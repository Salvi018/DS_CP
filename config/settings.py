"""Shared paths and runtime configuration for GPUlytics."""

import logging
from pathlib import Path

PROJECT_NAME = "GPUlytics"
PROJECT_VERSION = "0.1.0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"
REPORT_OUTPUT_DIR = PROJECT_ROOT / "reports"

DATASET_FILENAME = "disaggregated_DLRM_trace.csv"
DEFAULT_RANDOM_SEED = 42

LOGGING_CONFIG = {
    "level": logging.INFO,
    "format": "%(asctime)s %(levelname)s %(name)s: %(message)s",
}


def configure_logging() -> None:
    """Configure application-wide logging."""
    logging.basicConfig(**LOGGING_CONFIG)
