from config.settings import (
    DATASET_FILENAME,
    DEFAULT_RANDOM_SEED,
    MODEL_OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_NAME,
    PROJECT_VERSION,
    RAW_DATA_DIR,
    REPORT_OUTPUT_DIR,
)


def test_project_settings_use_project_relative_paths() -> None:
    assert PROJECT_NAME == "GPUlytics"
    assert PROJECT_VERSION
    assert DEFAULT_RANDOM_SEED == 42
    assert RAW_DATA_DIR == PROCESSED_DATA_DIR.parent / "raw"
    assert MODEL_OUTPUT_DIR.name == "models"
    assert REPORT_OUTPUT_DIR.name == "reports"
    assert DATASET_FILENAME == "disaggregated_DLRM_trace.csv"
