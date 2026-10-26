#!/usr/bin/env python3
"""Train and evaluate a historical scheduling-delay prediction pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DEFAULT_RANDOM_SEED,
    MODEL_OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    REPORT_OUTPUT_DIR,
)
from src.analytics import AnalyticsError, load_cleaned_dataset
from src.ml_training import TrainingDataError, train_delay_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train creation-time models for historical scheduling delay using "
            "a chronological train/validation/test split."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DATA_DIR / "cleaned_workloads.csv",
        help="Cleaned workload CSV (default: data/processed/cleaned_workloads.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORT_OUTPUT_DIR / "ml",
        help="Directory for model evaluation reports.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_OUTPUT_DIR,
        help="Directory for the serialized complete preprocessing/model pipeline.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random Forest seed.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Chronological fraction reserved for the final test set (0 < value < 0.5).",
    )
    parser.add_argument(
        "--model",
        choices=(
            "auto",
            "median_baseline",
            "dummy_median",
            "ridge",
            "random_forest",
        ),
        default="auto",
        help="Select by validation MAE (auto) or explicitly choose a model.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not 0 < args.test_size < 0.5:
        parser.error("--test-size must be greater than 0 and less than 0.5")
    try:
        frame = load_cleaned_dataset(args.input)
        result = train_delay_model(
            frame,
            dataset_filename=args.input.name,
            model_output_dir=args.model_dir,
            report_output_dir=args.output_dir,
            seed=args.seed,
            test_size=args.test_size,
            model=args.model,
        )
    except (AnalyticsError, TrainingDataError, OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Selected model: {result['selected_model']}")
    print(f"Serialized pipeline: {result['model_path']}")
    print(f"Report directory: {result['report_dir']}")
    print("Held-out test metrics:")
    for name, metrics in result["evaluation_metrics"]["test_metrics"].items():
        print(
            f"  {name}: n={metrics['sample_count']}, "
            f"MAE={metrics['mae']:.6g}, "
            f"MedAE={metrics['median_absolute_error']:.6g}, "
            f"RMSE={metrics['rmse']:.6g}, "
            f"R2={metrics['r_squared']}"
        )
    print("Target unit: unverified source timestamp units; not seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
