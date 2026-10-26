#!/usr/bin/env python3
"""Run the GPUlytics discrete-event simulator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DEFAULT_RANDOM_SEED, PROCESSED_DATA_DIR, REPORT_OUTPUT_DIR
from src.analytics import AnalyticsError, load_cleaned_dataset
from src.cluster_config import ClusterConfigError, DEFAULT_CLUSTER_CONFIG, load_cluster_config
from src.cluster_models import ResourceDemand, SimulatedWorkload
from src.schedulers import (
    DEFAULT_IFAS_CONFIG_PATH,
    IFASConfig,
    create_scheduler,
    load_ifas_config,
)
from src.simulation_comparison import compare_schedulers, save_comparison_report
from src.simulation_reports import save_simulation_report
from src.simulator import adapt_workload_frame, run_simulation, select_earliest_arrival_rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate trace workloads on a configurable hypothetical cluster "
            "using First Fit, Best Fit, or a controlled comparison."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DATA_DIR / "cleaned_workloads.csv",
        help="Cleaned workload CSV (default: data/processed/cleaned_workloads.csv).",
    )
    parser.add_argument(
        "--cluster-config",
        type=Path,
        default=DEFAULT_CLUSTER_CONFIG,
        help="JSON cluster configuration (default: config/demo_cluster.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORT_OUTPUT_DIR / "simulations",
        help="Directory for simulation reports.",
    )
    parser.add_argument(
        "--max-workloads",
        type=int,
        help="Limit to the earliest workloads with valid arrival times (for smoke tests).",
    )
    parser.add_argument(
        "--time-unit",
        default="trace_units_unverified",
        help="Label for source time values; no conversion is performed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Reproducibility metadata (the default policy is deterministic).",
    )
    policy_options = parser.add_mutually_exclusive_group()
    policy_options.add_argument(
        "--policy",
        choices=("first_fit", "best_fit", "ifas"),
        default="first_fit",
        help="Placement policy to run (default: first_fit).",
    )
    policy_options.add_argument(
        "--compare-policies",
        action="store_true",
        help="Run First Fit and Best Fit on identical inputs and write a comparison report.",
    )
    parser.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Run a clearly labeled in-memory demo instead of reading the CSV.",
    )
    parser.add_argument(
        "--ifas-config",
        type=Path,
        default=DEFAULT_IFAS_CONFIG_PATH,
        help="JSON weights/options for IFAS (default: config/ifas.json).",
    )
    return parser


def _synthetic_workloads() -> list[SimulatedWorkload]:
    """Small explicit demo inputs; not dataset-derived workload statistics."""
    return [
        SimulatedWorkload(
            "demo-cpu",
            ResourceDemand(4, 0, 16, 2, 20),
            role="CN",
            app_name="demo-cpu-app",
            arrival_time=0,
            duration=5,
        ),
        SimulatedWorkload(
            "demo-large-gpu",
            ResourceDemand(8, 8, 64, 4, 100),
            role="HN",
            app_name="demo-gpu-app",
            arrival_time=1,
            duration=3,
        ),
        SimulatedWorkload(
            "demo-small-gpu",
            ResourceDemand(8, 1, 32, 4, 100),
            role="HN",
            app_name="demo-gpu-app",
            arrival_time=2,
            duration=2,
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.max_workloads is not None and args.max_workloads < 1:
        parser.error("--max-workloads must be a positive integer")

    try:
        cluster = load_cluster_config(args.cluster_config)
        if args.synthetic_demo:
            workloads = _synthetic_workloads()
            rejected: list[dict[str, object]] = []
            source_row_count = len(workloads)
            input_name = "synthetic-demo (not dataset-derived)"
        else:
            frame = load_cleaned_dataset(args.input)
            if args.max_workloads is not None:
                frame = select_earliest_arrival_rows(frame, args.max_workloads)
            adapted = adapt_workload_frame(frame, cluster, time_unit=args.time_unit)
            cluster.assumptions.extend(
                [
                    f"Arrival source: {adapted.arrival_source or 'unavailable; affected rows are excluded'}.",
                    f"Duration source: {adapted.duration_source or 'unavailable; no completion durations are inferred'}.",
                    "Observed lifetimes are applied after simulated allocation; historical scheduling delays are not replayed.",
                ]
            )
            workloads = adapted.workloads
            rejected = adapted.rejected
            source_row_count = adapted.source_row_count
            input_name = str(args.input)
        if args.compare_policies:
            ifas_config = load_ifas_config(args.ifas_config)
            results = compare_schedulers(
                cluster,
                workloads,
                rejected_inputs=rejected,
                time_unit=args.time_unit,
                ifas_config=ifas_config,
            )
            files = save_comparison_report(
                results,
                cluster,
                args.output_dir / "comparison",
                input_filename=input_name,
                seed=args.seed,
                source_row_count=source_row_count,
                time_unit=args.time_unit,
                ifas_config=ifas_config,
            )
            for policy_name, result in results.items():
                print(
                    f"{policy_name}: {result.summary['allocated_workload_count']} allocated, "
                    f"{result.summary['completed_workload_count']} completed, "
                    f"{result.summary['pending_workload_count']} pending, "
                    f"{result.summary['rejected_workload_count']} rejected."
                )
        else:
            ifas_config = (
                load_ifas_config(args.ifas_config) if args.policy == "ifas" else None
            )
            scheduler = create_scheduler(args.policy, ifas_config)
            result = run_simulation(
                cluster,
                workloads,
                rejected_inputs=rejected,
                policy=scheduler,
                time_unit=args.time_unit,
            )
            files = save_simulation_report(
                result,
                cluster,
                args.output_dir / args.policy,
                input_filename=input_name,
                seed=args.seed,
                source_row_count=source_row_count,
            )
            print(
                f"{args.policy}: {result.summary['allocated_workload_count']} allocated, "
                f"{result.summary['completed_workload_count']} completed, "
                f"{result.summary['pending_workload_count']} pending, "
                f"{result.summary['rejected_workload_count']} rejected."
            )
    except (AnalyticsError, ClusterConfigError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print("Reports:")
    for file_path in files:
        print(f"  {file_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
