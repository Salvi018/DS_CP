#!/usr/bin/env python3
"""Generate workload analytics reports from the prepared dataset."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROCESSED_DATA_DIR, REPORT_OUTPUT_DIR
from src.analytics import AnalyticsError, run_analytics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calculate workload summaries and generate Plotly charts/reports."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DATA_DIR / "cleaned_workloads.csv",
        help=f"Cleaned workload CSV (default: {PROCESSED_DATA_DIR / 'cleaned_workloads.csv'}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORT_OUTPUT_DIR / "analytics",
        help=f"Analytics report directory (default: {REPORT_OUTPUT_DIR / 'analytics'}).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        result = run_analytics(args.input, args.output_dir)
    except (AnalyticsError, OSError, ValueError) as exc:
        print(f"Workload analytics failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"Analyzed {result.overview['row_count']:,} workloads from "
        f"{result.overview['dataset_filename']}."
    )
    print(f"Reports written to: {Path(args.output_dir).expanduser().resolve()}")
    if result.unavailable_analyses:
        print("Unavailable analyses: " + "; ".join(result.unavailable_analyses))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
