#!/usr/bin/env python3
"""Run reproducible scheduler benchmarks on synthetic or cleaned trace inputs."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DEFAULT_RANDOM_SEED, PROCESSED_DATA_DIR, REPORT_OUTPUT_DIR
from src.analytics import AnalyticsError, load_cleaned_dataset
from src.cluster_config import ClusterConfigError, DEFAULT_CLUSTER_CONFIG, load_cluster_config
from src.cluster_models import ResourceDemand, SimulatedWorkload
from src.evaluation import (
    SUPPORTED_POLICIES,
    run_experiment,
    run_sensitivity_analysis,
    save_benchmark_outputs,
    select_evaluation_rows,
)
from src.schedulers import load_ifas_config
from src.simulator import adapt_workload_frame


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare First Fit, Best Fit, and IFAS using identical workloads and "
            "fresh copies of the configured cluster."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DATA_DIR / "cleaned_workloads.csv",
        help="Cleaned workload CSV (default: data/processed/cleaned_workloads.csv).",
    )
    parser.add_argument(
        "--config",
        "--cluster-config",
        dest="config",
        type=Path,
        default=DEFAULT_CLUSTER_CONFIG,
        help="JSON cluster configuration (default: config/demo_cluster.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORT_OUTPUT_DIR / "benchmarks",
        help="Benchmark output root (an experiment-specific directory is created).",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=SUPPORTED_POLICIES,
        default=list(SUPPORTED_POLICIES),
        help="One or more policies (default: all three).",
    )
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument(
        "--max-workloads",
        type=int,
        help="Select the earliest valid-arrival workloads in trace order.",
    )
    subset.add_argument(
        "--sample-size",
        type=int,
        help="Select this many valid-arrival rows using the fixed --seed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Deterministic row-sampling seed and experiment metadata.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Requested repeat count; deterministic schedulers are executed once.",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run separate one-factor cluster and IFAS sensitivity scenarios.",
    )
    parser.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Benchmark a small explicit synthetic fixture rather than reading a CSV.",
    )
    parser.add_argument(
        "--time-unit",
        default="trace_units_unverified",
        help="Label for source time values; no conversion is performed.",
    )
    parser.add_argument(
        "--ifas-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "ifas.json",
        help="IFAS weights/options JSON (default: config/ifas.json).",
    )
    return parser


def _synthetic_workloads() -> list[SimulatedWorkload]:
    return [
        SimulatedWorkload(
            "benchmark-cpu",
            ResourceDemand(4, 0, 16, 2, 20),
            role="CN",
            app_name="synthetic-cpu-app",
            arrival_time=0,
            duration=5,
        ),
        SimulatedWorkload(
            "benchmark-gpu-large",
            ResourceDemand(8, 4, 64, 4, 100),
            role="HN",
            app_name="synthetic-gpu-app",
            arrival_time=1,
            duration=3,
        ),
        SimulatedWorkload(
            "benchmark-gpu-small",
            ResourceDemand(8, 1, 32, 4, 100),
            role="HN",
            app_name="synthetic-gpu-app",
            arrival_time=2,
            duration=2,
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be a positive integer")
    if args.max_workloads is not None and args.max_workloads < 1:
        parser.error("--max-workloads must be a positive integer")
    if args.sample_size is not None and args.sample_size < 1:
        parser.error("--sample-size must be a positive integer")
    if len(set(args.policies)) != len(args.policies):
        parser.error("--policies must not contain duplicates")

    try:
        cluster = load_cluster_config(args.config)
        selection_excluded: list[dict[str, object]] = []
        if args.synthetic_demo:
            workloads = _synthetic_workloads()
            rejected: list[dict[str, object]] = []
            source_row_count = len(workloads)
            selected_row_count = len(workloads)
            input_name = "synthetic-demo (not dataset-derived)"
        else:
            frame = load_cleaned_dataset(args.input)
            source_row_count = len(frame)
            selected, selection_excluded = select_evaluation_rows(
                frame,
                max_workloads=args.max_workloads,
                sample_size=args.sample_size,
                seed=args.seed,
            )
            selected_row_count = len(selected)
            adapted = adapt_workload_frame(
                selected, cluster, time_unit=args.time_unit
            )
            workloads = adapted.workloads
            rejected = adapted.rejected
            input_name = str(args.input)
            selection_excluded.extend(
                {
                    "source_row": item.get("source_row"),
                    "workload_id": item.get("workload_id"),
                    "reason": ";".join(item.get("reasons", [])),
                }
                for item in rejected
            )
            cluster.assumptions.extend(
                [
                    f"Workload arrival source: {adapted.arrival_source or 'missing; affected rows excluded'}.",
                    f"Duration source: {adapted.duration_source or 'unavailable; no completion times inferred'}.",
                    "Historical scheduling-delay values are not replayed; queue delay is measured only within the simulator.",
                ]
            )

        if not workloads:
            raise ValueError(
                "No valid workloads were selected. Check arrival, resource, role, "
                "and density-limit data-quality exclusions."
            )
        ifas_config = load_ifas_config(args.ifas_config)
        results = run_experiment(
            cluster,
            workloads,
            policies=args.policies,
            rejected_inputs=rejected,
            time_unit=args.time_unit,
            ifas_config=ifas_config,
        )
        sensitivity_results = (
            run_sensitivity_analysis(
                cluster,
                workloads,
                policies=args.policies,
                rejected_inputs=rejected,
                time_unit=args.time_unit,
                ifas_config=ifas_config,
            )
            if args.sensitivity
            else []
        )
        experiment_stamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        experiment_dir = args.output_dir / f"experiment-{experiment_stamp}"
        files = save_benchmark_outputs(
            results,
            cluster,
            workloads,
            experiment_dir,
            input_filename=input_name,
            source_row_count=source_row_count,
            selected_row_count=selected_row_count,
            seed=args.seed,
            time_unit=args.time_unit,
            ifas_config=ifas_config,
            selection_exclusions=selection_excluded,
            repeats_requested=args.repeats,
            sensitivity_results=sensitivity_results,
        )
        latest_report = args.output_dir / "benchmark_report.md"
        detailed_report = experiment_dir / "benchmark_report.md"
        latest_report.parent.mkdir(parents=True, exist_ok=True)
        latest_report.write_text(
            f"<!-- Latest experiment: {experiment_dir.name} -->\n\n"
            + detailed_report.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        files.append(latest_report)
    except (AnalyticsError, ClusterConfigError, OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    for policy_name, result in results.items():
        print(
            f"{policy_name}: {result.summary['allocated_workload_count']} allocated, "
            f"{result.summary['completed_workload_count']} completed, "
            f"{result.summary['pending_workload_count']} pending, "
            f"{result.summary['rejected_workload_count']} rejected, "
            f"{result.summary['resource_capacity_violation_count']} capacity violations."
        )
    print(f"Benchmark directory: {experiment_dir}")
    print(f"Latest benchmark report: {latest_report}")
    print("Reports:")
    for path in files:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
