#!/usr/bin/env python3
"""Validate, clean, and engineer features for a GPU workload trace."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROCESSED_DATA_DIR, REPORT_OUTPUT_DIR
from src.preprocessing import PreprocessingError, prepare_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a raw workload CSV, preserve auditable data-quality findings, "
            "and write a cleaned CSV plus cleaning reports."
        ),
        epilog=(
            "Warnings are non-fatal observations (for example missing timestamps or "
            "negative requests). Errors include invalid numeric values and essential "
            "identifier problems. --strict refuses to publish output when critical "
            "schema, numeric, or identifier errors are present."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Raw CSV path; defaults to the expected file or the only CSV in data/raw/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_DIR / "cleaned_workloads.csv",
        help=f"Cleaned CSV output (default: {PROCESSED_DATA_DIR / 'cleaned_workloads.csv'}).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORT_OUTPUT_DIR / "cleaning",
        help=f"Cleaning report directory (default: {REPORT_OUTPUT_DIR / 'cleaning'}).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on critical schema, ambiguous-column, and invalid-identifier/numeric errors.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        result = prepare_dataset(
            args.input,
            args.output,
            args.report_dir,
            strict=args.strict,
        )
    except (PreprocessingError, OSError, ValueError) as exc:
        print(f"Dataset preparation failed: {exc}", file=sys.stderr)
        return 2

    report = result.report
    print(
        f"Preparation {report['status']}: {report['input_row_count']:,} input rows, "
        f"{report['output_row_count']:,} output rows."
    )
    print(
        f"Validation findings: {report['warning_count']:,} warnings, "
        f"{report['error_count']:,} errors."
    )
    print(f"Cleaned dataset: {report['outputs']['cleaned_dataset']}")
    print(f"Cleaning report: {report['outputs']['cleaning_report_json']}")
    if report["excluded_rows"]:
        print(
            f"Excluded {report['excluded_rows']:,} rows; details: "
            f"{report['outputs']['rejected_rows_csv']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
