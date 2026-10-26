from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_explains_where_to_place_a_missing_dataset() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run()

    assert not app.exception
    dataset_path = app_path.parent / "data" / "raw" / "disaggregated_DLRM_trace.csv"
    if dataset_path.is_file():
        assert any("Dataset found" in message.value for message in app.success)
    else:
        assert any("Dataset not found" in warning.value for warning in app.warning)
