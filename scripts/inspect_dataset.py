#!/usr/bin/env python3
"""Run dataset inspection and exploratory analysis from the project root."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import REPORT_OUTPUT_DIR
from src.dataset_inspector import DatasetInspectionError, resolve_dataset_path
from src.eda import run_eda


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect an Alibaba GPU workload trace CSV and write schema, data-quality, "
            "summary, and optional chart reports."
        ),
        epilog=(
            "If --input is omitted, the expected data/raw CSV is used; otherwise a single "
            "CSV in data/raw may be selected automatically."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="CSV to inspect. Relative paths are resolved from the current directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORT_OUTPUT_DIR / "eda",
        help=f"Directory for generated reports (default: {REPORT_OUTPUT_DIR / 'eda'}).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write tabular/JSON/Markdown reports without generating charts.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50_000,
        help="Rows read per CSV chunk (default: 50000).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10_000,
        help="Maximum sampled values per column for quantiles and plots (default: 10000).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.chunksize < 1 or args.sample_size < 1:
        parser.error("--chunksize and --sample-size must be positive integers")
    try:
        dataset_path = resolve_dataset_path(args.input)
        inspection = run_eda(
            dataset_path,
            args.output_dir,
            skip_plots=args.skip_plots,
            chunksize=args.chunksize,
            sample_size=args.sample_size,
        )
    except DatasetInspectionError as exc:
        print(f"Dataset inspection error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"Could not write report files: {exc}", file=sys.stderr)
        return 2

    report = inspection.report
    print(
        f"Inspected {report['dataset_filename']}: "
        f"{report['row_count']:,} rows × {report['column_count']} columns."
    )
    print(f"Reports written to: {Path(args.output_dir).expanduser().resolve()}")
    if report["generated_charts"]:
        print("Charts generated: " + ", ".join(report["generated_charts"]))
    if report["missing_expected_columns"]:
        print("Missing expected columns: " + ", ".join(report["missing_expected_columns"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
