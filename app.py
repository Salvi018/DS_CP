"""Streamlit entry point for the GPUlytics dashboard."""

import json
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

from config.settings import (
    DATASET_FILENAME,
    MODEL_OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    REPORT_OUTPUT_DIR,
)
from src.dataset_inspector import DatasetInspectionError, inspect_dataset, resolve_dataset_path
from src.preprocessing import PreprocessingError, prepare_dataset
from src.analytics import (
    AnalyticsError,
    analyze_workloads,
    create_analytics_figures,
    load_cleaned_dataset,
)
from src.cluster_config import ClusterConfigError, load_cluster_config
from src.schedulers import IFASConfig, create_scheduler
from src.simulation_comparison import (
    compare_schedulers,
    comparison_metrics,
    save_comparison_report,
)
from src.simulation_reports import save_simulation_report
from src.evaluation import (
    SUPPORTED_POLICIES,
    run_experiment,
    run_sensitivity_analysis,
    save_benchmark_outputs,
    select_evaluation_rows,
)
from src.ml_prediction import PredictionError, load_delay_model, predict_scheduling_delay
from src.ml_training import TrainingDataError, train_delay_model
from src.dashboard_data import (
    available_filter_columns,
    filter_workloads,
    latest_benchmark_dir,
    list_report_artifacts,
    load_report_artifact,
    report_source_freshness,
)
from src.simulator import (
    adapt_workload_frame,
    run_simulation,
    select_earliest_arrival_rows,
)


@st.cache_data(show_spinner=False)
def _cached_inspection(path: str, file_size: int, modified_ns: int):
    del file_size, modified_ns
    return inspect_dataset(path)


@st.cache_data(show_spinner=False)
def _read_json_report(path: str, file_size: int, modified_ns: int):
    del file_size, modified_ns
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


@st.cache_data(show_spinner=False)
def _cached_cleaned_frame(path: str, file_size: int, modified_ns: int) -> pd.DataFrame:
    del file_size, modified_ns
    return load_cleaned_dataset(path)


@st.cache_data(show_spinner=False)
def _cached_analytics(
    path: str,
    file_size: int,
    modified_ns: int,
    filters_json: str = "{}",
):
    filters = json.loads(filters_json)
    filtered = filter_workloads(
        _cached_cleaned_frame(path, file_size, modified_ns),
        categories=filters.get("categories", {}),
        numeric_ranges={
            column: tuple(bounds)
            for column, bounds in filters.get("numeric_ranges", {}).items()
        },
    )
    result = analyze_workloads(filtered, dataset_filename=Path(path).name)
    figures = create_analytics_figures(result.classified_workloads)
    return result, figures


def _read_cached_report(path: Path) -> dict | None:
    try:
        if not path.is_file():
            return None
        stat = path.stat()
    except OSError:
        return None
    try:
        return _read_json_report(str(path), stat.st_size, stat.st_mtime_ns)
    except (OSError, json.JSONDecodeError):
        return None


def _read_report_csv(path: Path) -> pd.DataFrame | None:
    artifact = load_report_artifact(path)
    if artifact.kind != "csv":
        st.warning(
            artifact.error
            or f"Optional report is unavailable: `{path}`."
        )
        return None
    return artifact.content


def main() -> None:
    st.set_page_config(page_title="GPUlytics", page_icon="🖥️", layout="wide")
    pio.templates.default = "plotly_dark"
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
            --gpu-ink: #e9f0f6;
            --gpu-muted: #9eafbd;
            --gpu-panel: #121e29;
            --gpu-border: #29404f;
            --gpu-accent: #3ed6b0;
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(ellipse at 78% -8%, rgba(30, 103, 98, .22), transparent 35rem),
                linear-gradient(160deg, #0b131c 0%, #101b25 52%, #0b141d 100%);
            color: var(--gpu-ink);
        }
        [data-testid="stHeader"] { background: rgba(11, 19, 28, .84); }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111f2a 0%, #0d1720 100%);
            border-right: 1px solid var(--gpu-border);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--gpu-muted);
        }
        [data-testid="stMetric"] {
            background: rgba(18, 30, 41, .84);
            border: 1px solid var(--gpu-border);
            border-left: 3px solid var(--gpu-accent);
            padding: .8rem 1rem;
        }
        [data-testid="stMetricLabel"] { color: var(--gpu-muted); }
        [data-testid="stMetricValue"] { color: var(--gpu-ink); }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border: 1px solid var(--gpu-border);
        }
        div[data-testid="stExpander"] {
            border: 1px solid var(--gpu-border);
            background: rgba(18, 30, 41, .48);
        }
        div.stButton > button, div.stDownloadButton > button {
            border-radius: 3px;
            border: 1px solid #3f766f;
            color: #e8fff8;
            background: #17372f;
            transition: background-color .16s ease, border-color .16s ease;
        }
        div.stButton > button:hover, div.stDownloadButton > button:hover {
            background: #205045;
            border-color: var(--gpu-accent);
        }
        div.stButton > button:focus-visible, div.stDownloadButton > button:focus-visible {
            outline: 2px solid var(--gpu-accent);
            outline-offset: 2px;
        }
        h1, h2, h3 { letter-spacing: -.025em; }
        a { color: var(--gpu-accent) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("GPUlytics")
    st.subheader("Intelligent GPU Monitoring and Resource Management")
    st.write(
        "Explore GPU workload data, simulate cluster scheduling, and compare "
        "resource allocation strategies."
    )

    sections = [
        "Overview",
        "Dataset Inspection",
        "Workload Analytics",
        "Data Quality & Reports",
        "Data Preparation",
        "Cluster Simulator",
        "Scheduler Comparison",
        "AI Predictions",
        "Algorithm Comparison",
    ]
    selected_section = st.sidebar.radio("Navigation", sections)

    st.sidebar.markdown("**GPUlytics**")
    if selected_section == "Overview":
        _render_overview()
        return
    if selected_section == "Dataset Inspection":
        _render_dataset_inspection()
        return
    if selected_section == "Workload Analytics":
        _render_workload_analytics()
        return
    if selected_section == "Data Quality & Reports":
        _render_data_quality_reports()
        return
    if selected_section == "Data Preparation":
        _render_data_preparation()
        return
    if selected_section == "Cluster Simulator":
        _render_cluster_simulator()
        return
    if selected_section == "Scheduler Comparison":
        _render_benchmarking()
        return
    if selected_section == "AI Predictions":
        _render_ai_predictions()
        return
    if selected_section == "Algorithm Comparison":
        _render_algorithm_comparison()
        return
    st.header(selected_section)
    st.info(f"{selected_section} is planned and has not been implemented yet.")


def _render_dataset_inspection() -> None:
    st.header("Dataset Inspection")
    try:
        dataset_path = resolve_dataset_path()
    except DatasetInspectionError as exc:
        st.error(str(exc))
        st.info(
            "Download the Alibaba GPU Cluster Dataset 2025 and place the CSV at "
            f"`data/raw/{DATASET_FILENAME}`. If you have another CSV in `data/raw/`, "
            "the command-line inspector can use it when it is the only candidate."
        )
        return

    st.success(f"Dataset found: `{dataset_path.name}`")
    st.caption(f"Selected CSV: `{dataset_path}`")
    if st.button("Refresh inspection"):
        _cached_inspection.clear()
    try:
        stat = dataset_path.stat()
        with st.spinner("Inspecting dataset in chunks..."):
            inspection = _cached_inspection(
                str(dataset_path), stat.st_size, stat.st_mtime_ns
            )
    except (DatasetInspectionError, OSError) as exc:
        st.error(f"Dataset inspection failed: {exc}")
        return

    first, second = st.columns(2)
    first.metric("Rows", f"{inspection.report.get('row_count', 0):,}")
    second.metric("Columns", inspection.report.get("column_count", 0))

    if hasattr(inspection, "sample") and isinstance(inspection.sample, pd.DataFrame) and not inspection.sample.empty:
        st.subheader("Dataset sample preview")
        st.dataframe(inspection.sample.head(20), width="stretch", hide_index=True)

    st.subheader("Detected columns and types")
    st.dataframe(inspection.schema, width="stretch", hide_index=True)

    if hasattr(inspection, "numerical_summary") and isinstance(inspection.numerical_summary, pd.DataFrame) and not inspection.numerical_summary.empty:
        st.subheader("Numerical column statistics")
        st.dataframe(inspection.numerical_summary, width="stretch", hide_index=True)

    if hasattr(inspection, "categorical_summary") and isinstance(inspection.categorical_summary, pd.DataFrame) and not inspection.categorical_summary.empty:
        st.subheader("Categorical column summary")
        st.dataframe(inspection.categorical_summary, width="stretch", hide_index=True)

    st.subheader("Missing values")
    if hasattr(inspection, "missing_values") and isinstance(inspection.missing_values, pd.DataFrame) and not inspection.missing_values.empty and "missing_count" in inspection.missing_values.columns:
        missing = inspection.missing_values.loc[inspection.missing_values["missing_count"] > 0]
    else:
        missing = pd.DataFrame()
    if missing.empty:
        st.success("No missing values detected.")
    else:
        st.dataframe(missing, width="stretch", hide_index=True)

    st.subheader("Data quality")
    if inspection.data_quality.empty:
        st.success("No data-quality warnings were detected.")
    else:
        st.dataframe(inspection.data_quality, width="stretch", hide_index=True)

    reports_dir = REPORT_OUTPUT_DIR / "eda"
    st.subheader("Generated reports")
    if reports_dir.is_dir() and any(reports_dir.iterdir()):
        for report_file in sorted(reports_dir.iterdir()):
            if report_file.is_file():
                st.caption(str(report_file))
    else:
        st.info(
            "No saved EDA reports yet. Run "
            "`python scripts/inspect_dataset.py` from the project directory to generate them."
        )


def _render_overview() -> None:
    st.header("GPU workload operations")
    st.caption(
        "A trace-grounded view of the prepared dataset, analytical summaries, "
        "and available model artifacts. No simulation, benchmark, or training "
        "runs automatically from this page."
    )
    raw_path = None
    try:
        raw_path = resolve_dataset_path()
    except DatasetInspectionError:
        pass
    cleaned_path = PROCESSED_DATA_DIR / "cleaned_workloads.csv"
    cleaning_path = REPORT_OUTPUT_DIR / "cleaning" / "cleaning_report.json"
    model_path = MODEL_OUTPUT_DIR / "scheduling_delay_pipeline.joblib"
    cleaning_report = _read_cached_report(cleaning_path)
    cleaning_freshness = (
        report_source_freshness(cleaning_report, raw_path)
        if isinstance(cleaning_report, dict) and raw_path is not None
        else "unknown"
    )

    status_columns = st.columns(4)
    status_columns[0].metric(
        "Raw source",
        "Available" if raw_path is not None else "Not found",
    )
    status_columns[1].metric(
        "Prepared dataset",
        "Available" if cleaned_path.is_file() else "Not found",
    )
    status_columns[2].metric(
        "Cleaning report",
        {
            "current": "Current",
            "stale": "Stale",
            "source_unavailable": "Source unavailable",
            "unknown": "Freshness unknown",
        }.get(cleaning_freshness, "Unavailable")
        if cleaning_report is not None
        else "Unavailable",
    )
    status_columns[3].metric(
        "Delay model artifact",
        "Available" if model_path.is_file() else "Not trained",
    )
    if raw_path is not None:
        st.success(f"Dataset found: `{raw_path}`")
        st.caption(f"Raw source: `{raw_path}`")
    else:
        st.info(
            f"Raw dataset not found. Expected `{RAW_DATA_DIR / DATASET_FILENAME}`."
        )

    st.subheader("Project module status")
    module_status = {
        "Workload analytics": PROJECT_ROOT / "src" / "analytics.py",
        "Cluster simulator": PROJECT_ROOT / "src" / "simulator.py",
        "Scheduler evaluation": PROJECT_ROOT / "src" / "evaluation.py",
        "Delay inference": PROJECT_ROOT / "src" / "ml_prediction.py",
    }
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "module": name,
                    "status": "Available" if path.is_file() else "Missing",
                    "path": str(path.relative_to(PROJECT_ROOT)),
                }
                for name, path in module_status.items()
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    if cleaning_report:
        if cleaning_freshness == "stale":
            st.warning(
                "The cleaning report source fingerprint does not match the current "
                "raw dataset. Its counts describe an older/different input; rerun "
                "preprocessing to refresh the report."
            )
        elif cleaning_freshness == "unknown":
            st.info(
                "The cleaning report does not include a complete source fingerprint; "
                "its freshness cannot be verified."
            )
        dispositions = st.columns(5)
        for column, label, key in zip(
            dispositions,
            (
                "Input records",
                "Retained records",
                "Excluded records",
                "Validation warnings",
                "Validation errors",
            ),
            (
                "input_row_count",
                "output_row_count",
                "excluded_rows",
                "warning_count",
                "error_count",
            ),
        ):
            value = cleaning_report.get(key)
            column.metric(
                label,
                f"{value:,}" if isinstance(value, int) else "Unavailable",
            )
        st.caption(
            "Warning and error totals are validation findings, not counts of "
            "unique invalid records. Retained rows may still contain missing "
            "timestamps or resource values."
        )
    elif cleaning_path.exists():
        st.warning("The saved cleaning report is malformed or unreadable.")
    else:
        st.info("No saved cleaning report is available.")

    if not cleaned_path.is_file():
        st.warning(
            "Cleaned workload analytics are unavailable. Use Data Preparation "
            "to explicitly create the prepared dataset."
        )
        return
    try:
        stat = cleaned_path.stat()
        frame = _cached_cleaned_frame(str(cleaned_path), stat.st_size, stat.st_mtime_ns)
        analytics = analyze_workloads(frame, dataset_filename=cleaned_path.name)
    except (AnalyticsError, OSError, ValueError) as exc:
        st.error(f"Could not read cleaned workload summary: {exc}")
        return

    st.subheader("Prepared workload snapshot")
    st.caption(f"Source file: `{cleaned_path}`")
    summary = analytics.overview
    count_columns = st.columns(4)
    count_columns[0].metric("Cleaned records", f"{summary['row_count']:,}")
    app_groups = summary.get("application_group_count")
    count_columns[1].metric(
        "Application groups",
        f"{app_groups:,}" if isinstance(app_groups, int) else "Unavailable",
    )
    gpu_count = summary.get("gpu_requesting_workload_count")
    count_columns[2].metric(
        "GPU-requesting records",
        f"{gpu_count:,}" if isinstance(gpu_count, int) else "Unavailable",
    )
    count_columns[3].metric(
        "Columns",
        f"{summary['column_count']:,}",
    )

    if not analytics.role_summary.empty:
        st.subheader("Workload distribution")
        st.dataframe(analytics.role_summary, width="stretch", hide_index=True)
    else:
        st.info("Role breakdown unavailable: the cleaned dataset has no role data.")
    if not analytics.resource_summary.empty:
        st.subheader("Resource-request breakdown")
        requests = analytics.resource_summary.loc[
            analytics.resource_summary["metric_kind"] == "requested"
        ]
        if requests.empty:
            st.info("No resource request columns are available.")
        else:
            st.dataframe(requests, width="stretch", hide_index=True)
    else:
        st.info("Resource-request breakdown unavailable for this dataset schema.")
    st.caption(summary["resource_metric_note"])

    st.subheader("Creation-time trend")
    if "creation_time" not in frame.columns:
        st.info(
            "Creation-time trend unavailable: the cleaned dataset has no "
            "`creation_time` column."
        )
    else:
        creation = pd.to_numeric(frame["creation_time"], errors="coerce")
        valid_creation = creation[creation.map(math.isfinite)].dropna()
        if valid_creation.empty:
            st.info(
                "Creation-time trend unavailable: no finite numeric creation "
                "timestamps are present."
            )
        else:
            trend = pd.DataFrame({"creation_time": valid_creation})
            fig = px.histogram(
                trend,
                x="creation_time",
                nbins=40,
                title="Observed workload arrivals across creation-time values",
                labels={
                    "creation_time": "Creation time (raw source units; unverified)",
                    "count": "Observed records",
                },
                color_discrete_sequence=["#3ed6b0"],
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                f"{len(valid_creation):,} finite timestamps shown. Source units are "
                "unverified and are not converted to seconds."
            )


def _render_data_preparation() -> None:
    st.header("Data Preparation")
    st.write(
        "Prepare a cleaned workload CSV with an audit trail. Preparation only runs "
        "after you select the button below."
    )
    expected_raw = RAW_DATA_DIR / DATASET_FILENAME
    cleaned_path = PROJECT_ROOT / "data" / "processed" / "cleaned_workloads.csv"
    cleaning_report_path = REPORT_OUTPUT_DIR / "cleaning" / "cleaning_report.json"
    st.caption(f"Expected raw CSV: `{expected_raw}`")
    st.caption(f"Cleaned CSV: `{cleaned_path}`")
    st.caption(f"Cleaning report: `{cleaning_report_path}`")

    try:
        raw_path = resolve_dataset_path()
    except DatasetInspectionError as exc:
        raw_path = None
        st.error(str(exc))
    if raw_path is not None:
        st.success(f"Raw dataset available: `{raw_path.name}`")
    else:
        st.warning("No unambiguous raw dataset is available for preparation.")

    prior_report = _read_cached_report(cleaning_report_path)
    raw_rows = None
    if prior_report and raw_path:
        raw_stat = raw_path.stat()
        if (
            prior_report.get("input_path") == str(raw_path)
            and prior_report.get("input_size_bytes") == raw_stat.st_size
            and prior_report.get("input_mtime_ns") == raw_stat.st_mtime_ns
        ):
            raw_rows = prior_report.get("input_row_count")

    cleaned_rows = None
    if cleaned_path.is_file() and prior_report:
        if prior_report.get("outputs", {}).get("cleaned_dataset") == str(cleaned_path.resolve()):
            cleaned_rows = prior_report.get("output_row_count")
    st.metric("Raw rows (when known)", f"{raw_rows:,}" if raw_rows is not None else "Not recorded")
    st.metric(
        "Cleaned rows (when known)",
        f"{cleaned_rows:,}" if cleaned_rows is not None else ("Present; count not recorded" if cleaned_path.is_file() else "Not available"),
    )
    if cleaned_path.is_file():
        st.success("Cleaned dataset is available.")
    else:
        st.info("No cleaned dataset is available yet.")

    if prior_report:
        st.subheader("Last preparation status")
        st.write(
            f"Status: **{prior_report.get('status', 'unknown')}** · "
            f"Prepared: {prior_report.get('prepared_at_utc', 'not recorded')}"
        )
        st.write(
            f"Validation findings: {prior_report.get('warning_count', 0):,} warnings, "
            f"{prior_report.get('error_count', 0):,} errors."
        )
        st.caption(f"Cleaning audit: `{cleaning_report_path}`")
    else:
        st.info("No previous cleaning report is available.")

    if st.button("Prepare dataset", disabled=raw_path is None):
        try:
            with st.spinner("Validating and preparing workload data..."):
                result = prepare_dataset()
        except (PreprocessingError, OSError, ValueError) as exc:
            st.error(f"Dataset preparation failed: {exc}")
            st.info("Any previously prepared dataset is preserved when preparation fails.")
        else:
            _read_json_report.clear()
            _cached_cleaned_frame.clear()
            _cached_analytics.clear()
            st.success(
                f"Preparation {result.report['status']}: "
                f"{result.report['input_row_count']:,} input rows, "
                f"{result.report['output_row_count']:,} output rows."
            )
            st.write(
                f"Warnings: {result.report['warning_count']:,} · "
                f"Errors: {result.report['error_count']:,}"
            )
            st.caption(f"Cleaning report: `{cleaning_report_path}`")

    if cleaning_report_path.is_file():
        st.subheader("Download preparation report")
        try:
            st.download_button(
                "Download cleaning_report.json",
                data=cleaning_report_path.read_bytes(),
                file_name="cleaning_report.json",
                mime="application/json",
                key="download_cleaning_report_json",
            )
        except OSError:
            pass
        summary_md_path = REPORT_OUTPUT_DIR / "cleaning" / "cleaning_summary.md"
        if summary_md_path.is_file():
            try:
                st.download_button(
                    "Download cleaning_summary.md",
                    data=summary_md_path.read_bytes(),
                    file_name="cleaning_summary.md",
                    mime="text/markdown",
                    key="download_cleaning_summary_md",
                )
            except OSError:
                pass


def _render_workload_analytics() -> None:
    st.header("Workload Analytics")
    cleaned_path = PROCESSED_DATA_DIR / "cleaned_workloads.csv"
    if not cleaned_path.is_file():
        st.warning(
            "The cleaned workload dataset is not available. Run "
            "`python scripts/prepare_dataset.py` from the project directory first."
        )
        return
    st.caption(f"Dataset: `{cleaned_path}`")
    try:
        stat = cleaned_path.stat()
        frame = _cached_cleaned_frame(
            str(cleaned_path), stat.st_size, stat.st_mtime_ns
        )
    except (AnalyticsError, OSError, ValueError) as exc:
        st.error(f"Could not load workload analytics data: {exc}")
        return
    category_columns, numeric_columns = available_filter_columns(frame)
    st.subheader("Filter available workload columns")
    categories: dict[str, list[str]] = {}
    numeric_ranges: dict[str, tuple[float, float]] = {}
    filter_reset_version = st.session_state.get(
        "gpulytics_analytics_filter_reset", 0
    )
    selected_numeric = st.multiselect(
        "Numeric columns to constrain",
        options=numeric_columns,
        default=[],
        format_func=lambda column: column.replace("_", " ").title(),
        help="Only numeric columns present in this cleaned dataset are offered. Selecting a column reveals its range slider below.",
        key=f"analytics_num_cols_{filter_reset_version}",
    )
    with st.form("workload_analytics_filters"):
        category_controls = st.columns(max(1, len(category_columns)))
        for column, control in zip(category_columns, category_controls):
            options = sorted(
                str(value)
                for value in frame[column].dropna().unique()
                if str(value).strip()
            )
            selected = control.multiselect(
                f"{column.replace('_', ' ').title()} (all if empty)",
                options=options,
                key=f"analytics_filter_{filter_reset_version}_{column}",
            )
            if selected:
                categories[column] = selected
        for column in selected_numeric:
            values = pd.to_numeric(frame[column], errors="coerce")
            values = values[values.map(math.isfinite)].dropna()
            if values.empty:
                st.caption(f"{column}: no finite values are available to filter.")
                continue
            lower, upper = float(values.min()), float(values.max())
            if lower == upper:
                bounds = (lower, upper)
                st.caption(f"{column}: only {lower:g} is available.")
            else:
                bounds = st.slider(
                    f"{column.replace('_', ' ').title()} range",
                    min_value=lower,
                    max_value=upper,
                    value=(lower, upper),
                    key=f"analytics_range_{filter_reset_version}_{column}",
                )
            numeric_ranges[column] = bounds
        apply_filters = st.form_submit_button("Apply filters")
    if apply_filters:
        st.session_state["gpulytics_analytics_filters"] = {
            "categories": categories,
            "numeric_ranges": numeric_ranges,
        }
    if st.button("Clear filters"):
        st.session_state.pop("gpulytics_analytics_filters", None)
        st.session_state["gpulytics_analytics_filter_reset"] = (
            filter_reset_version + 1
        )
        st.rerun()
    active_filters = st.session_state.get(
        "gpulytics_analytics_filters",
        {"categories": {}, "numeric_ranges": {}},
    )
    filtered_frame = filter_workloads(
        frame,
        categories=active_filters.get("categories", {}),
        numeric_ranges=active_filters.get("numeric_ranges", {}),
    )
    if st.button("Refresh analytics"):
        _cached_analytics.clear()
        _cached_cleaned_frame.clear()
    try:
        stat = cleaned_path.stat()
        with st.spinner("Calculating workload analytics..."):
            result, figures = _cached_analytics(
                str(cleaned_path),
                stat.st_size,
                stat.st_mtime_ns,
                json.dumps(active_filters, sort_keys=True),
            )
    except (AnalyticsError, OSError, ValueError) as exc:
        st.error(f"Could not calculate workload analytics: {exc}")
        return
    st.caption(
        f"Showing {len(filtered_frame):,} of {len(frame):,} source rows. "
        "Missing or invalid values do not match an active numeric range."
    )
    if filtered_frame.empty:
        st.warning("No workloads match the selected filter criteria. Showing empty results.")

    overview = result.overview
    first, second, third = st.columns(3)
    first.metric("Workloads", f"{overview['row_count']:,}")
    application_count = overview["application_group_count"]
    second.metric(
        "Application groups",
        f"{application_count:,}" if application_count is not None else "Unavailable",
    )
    gpu_count = overview["gpu_requesting_workload_count"]
    third.metric(
        "GPU-requesting workloads",
        f"{gpu_count:,}" if gpu_count is not None else "Unavailable",
    )

    st.caption(overview["resource_metric_note"])
    if not result.role_summary.empty:
        st.subheader("Workload counts by role")
        st.dataframe(result.role_summary, width="stretch", hide_index=True)
    if not result.resource_summary.empty:
        st.subheader("Resource request and limit summaries")
        st.dataframe(
            result.resource_summary,
            width="stretch",
            hide_index=True,
        )
        request_summaries = result.resource_summary[
            result.resource_summary["metric_kind"] == "requested"
        ]
        if not request_summaries.empty:
            resource_columns = st.columns(max(1, min(5, len(request_summaries))))
            for column, (_, summary) in zip(resource_columns, request_summaries.iterrows()):
                mean = summary["mean"]
                column.metric(
                    f"{str(summary.get('column', '')).split('_')[0].upper()} request mean",
                    f"{mean:,.3f}"
                    if isinstance(mean, (int, float)) and math.isfinite(mean)
                    else "Unavailable",
                )
    if not result.application_summary.empty:
        st.subheader("Top applications")
        st.dataframe(
            result.application_summary.head(20),
            width="stretch",
            hide_index=True,
        )
    st.subheader("Workload size heuristic")
    st.caption(overview["workload_size_threshold_note"])
    st.dataframe(result.workload_size_summary, width="stretch", hide_index=True)

    for name, figure in figures.items():
        st.subheader(name.replace("_", " ").title())
        st.plotly_chart(figure, width="stretch", key=f"analytics_{name}")

    if result.scheduling_delay_summary is not None:
        st.subheader("Scheduling delay (unverified source units)")
        delay = result.scheduling_delay_summary
        metrics = st.columns(5)
        for column, label, key in zip(
            metrics,
            ("Mean", "Median", "P95", "P99", "Negative observations"),
            ("mean", "median", "p95", "p99", "negative_count"),
        ):
            value = delay.get(key)
            column.metric(label, f"{value:,.3f}" if isinstance(value, (int, float)) and value is not None else "Unavailable")
        st.caption(
            f"Missing observations: {delay['missing_count']:,}; "
            f"invalid observations: {delay['invalid_count']:,}."
        )
    if result.lifetime_summary is not None:
        st.subheader("Observed scheduled lifetime (unverified source units)")
        lifetime = result.lifetime_summary
        st.write(
            f"Mean: {lifetime['mean'] if lifetime['mean'] is not None else 'Unavailable'} · "
            f"Median: {lifetime['median'] if lifetime['median'] is not None else 'Unavailable'} · "
            f"p95: {lifetime['p95'] if lifetime['p95'] is not None else 'Unavailable'} · "
            f"p99: {lifetime['p99'] if lifetime['p99'] is not None else 'Unavailable'} · "
            f"Negative: {lifetime['negative_count']:,} · "
            f"Missing: {lifetime['missing_count']:,}"
        )
        st.caption(overview["observed_lifetime_limitation"])
    if not result.gpu_comparison.empty:
        st.subheader("GPU-requesting versus zero-GPU-request workloads")
        st.dataframe(result.gpu_comparison, width="stretch", hide_index=True)
    if not result.data_quality_summary.empty:
        st.subheader("Data-quality findings")
        st.dataframe(result.data_quality_summary, width="stretch", hide_index=True)
    if result.unavailable_analyses:
        st.subheader("Unavailable analyses")
        for name, reason in result.unavailable_analyses.items():
            st.caption(f"{name}: {reason}")
    st.caption(
        f"Saved analytics reports (when generated by the analytics CLI): "
        f"`{REPORT_OUTPUT_DIR / 'analytics'}`"
    )


def _render_data_quality_reports() -> None:
    st.header("Data Quality & Reports")
    cleaned_path = PROCESSED_DATA_DIR / "cleaned_workloads.csv"
    cleaning_report_path = REPORT_OUTPUT_DIR / "cleaning" / "cleaning_report.json"
    st.caption(f"Prepared dataset: `{cleaned_path}`")
    if cleaned_path.is_file():
        st.success("Prepared dataset is present.")
    else:
        st.warning(
            "Prepared dataset is missing. Run `python scripts/prepare_dataset.py` "
            "only when you explicitly want to create or refresh it."
        )

    cleaning_report = _read_cached_report(cleaning_report_path)
    if isinstance(cleaning_report, dict):
        st.subheader("Saved preparation quality summary")
        try:
            current_raw_path = resolve_dataset_path()
        except DatasetInspectionError:
            current_raw_path = None
        freshness = (
            report_source_freshness(cleaning_report, current_raw_path)
            if current_raw_path is not None
            else "source_unavailable"
        )
        if freshness == "stale":
            st.warning(
                "This report was generated from a different or modified raw file. "
                "The counts below are historical report values, not a summary of "
                "the currently available source."
            )
        elif freshness in {"unknown", "source_unavailable"}:
            st.info(
                "Report freshness cannot be confirmed because the source file or "
                "its recorded fingerprint is unavailable."
            )
        disposition_columns = st.columns(5)
        for column, label, key in zip(
            disposition_columns,
            (
                "Input records",
                "Retained",
                "Excluded",
                "Warning findings",
                "Error findings",
            ),
            (
                "input_row_count",
                "output_row_count",
                "excluded_rows",
                "warning_count",
                "error_count",
            ),
        ):
            value = cleaning_report.get(key)
            column.metric(
                label,
                f"{value:,}" if isinstance(value, int) else "Unavailable",
            )
        st.caption(
            f"Preparation status: {cleaning_report.get('status', 'Unavailable')}"
        )
        st.caption(
            cleaning_report.get(
                "timestamp_unit_assumption",
                "Timestamp-unit assumption is not recorded in this report.",
            )
        )
        issue_counts = cleaning_report.get("issue_counts")
        if isinstance(issue_counts, dict) and issue_counts:
            st.subheader("Recorded issue counts")
            st.dataframe(
                pd.DataFrame(
                    [{"issue": name, "count": count} for name, count in issue_counts.items()]
                ),
                width="stretch",
                hide_index=True,
            )
    elif cleaning_report_path.exists():
        st.warning("The saved preparation report is malformed or unreadable.")
    else:
        st.info("No preparation report is available yet.")

    st.info(
        "Data limitations: source timestamp units and conventions are unverified "
        "and are not converted to seconds. `max_instance_per_node` semantics are "
        "also unverified; negative values are not assumed to mean unlimited, and "
        "the configured simulator does not enforce density limits unless their "
        "semantics are explicitly verified."
    )

    st.subheader("Saved report artifacts")
    artifacts = list_report_artifacts(REPORT_OUTPUT_DIR)
    if not artifacts:
        st.info("No saved report artifacts are available.")
        return
    labels = {
        str(path): str(path.relative_to(PROJECT_ROOT))
        for path in artifacts
    }
    selected_path = st.selectbox(
        "Choose an existing report",
        options=list(labels),
        format_func=lambda path: labels[path],
    )
    artifact = load_report_artifact(selected_path)
    selected_artifact = Path(selected_path)
    download_mime_types = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".md": "text/markdown",
        ".txt": "text/plain",
        ".html": "text/html",
        ".htm": "text/html",
    }
    if selected_artifact.suffix.lower() in download_mime_types:
        try:
            artifact_size = selected_artifact.stat().st_size
            if artifact_size <= 25 * 1024 * 1024:
                st.download_button(
                    "Download selected artifact",
                    data=selected_artifact.read_bytes(),
                    file_name=selected_artifact.name,
                    mime=download_mime_types[selected_artifact.suffix.lower()],
                    key=f"download_report_{selected_artifact.as_posix()}",
                )
            else:
                st.caption(
                    "Download is disabled for selected reports larger than 25 MiB."
                )
        except OSError as exc:
            st.warning(f"Could not prepare report download: {exc}")
    if artifact.kind in {"error", "missing"}:
        st.warning(artifact.error or "This artifact is unavailable.")
    elif artifact.kind == "json":
        st.json(artifact.content)
    elif artifact.kind == "csv":
        if artifact.content.empty:
            st.info("This report contains headers but no data rows.")
        else:
            st.dataframe(artifact.content, width="stretch", hide_index=True)
    elif artifact.kind == "text":
        st.markdown(artifact.content)
    else:
        st.info(artifact.content)


def _render_cluster_simulator() -> None:
    st.header("Cluster Simulator")
    cleaned_path = PROCESSED_DATA_DIR / "cleaned_workloads.csv"
    cluster_config_path = PROJECT_ROOT / "config" / "demo_cluster.json"
    output_dir = REPORT_OUTPUT_DIR / "simulations"
    st.warning(
        "This is a deterministic simulator on a hypothetical cluster. Resource "
        "requests are reservations in configured request units, not measured "
        "hardware utilization. Timestamp units are unverified."
    )
    if not cleaned_path.is_file():
        st.info(
            "The cleaned workload dataset is unavailable. Run "
            "`python scripts/prepare_dataset.py` from the project directory first."
        )
        return

    st.caption(f"Cleaned dataset: `{cleaned_path}`")
    st.caption(f"Cluster configuration: `{cluster_config_path}`")
    st.caption(f"Simulation reports: `{output_dir}`")
    try:
        cluster_preview = load_cluster_config(cluster_config_path)
    except ClusterConfigError as exc:
        st.error(f"Cluster configuration could not be loaded: {exc}")
        return
    with st.expander("Configured hypothetical cluster and assumptions"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "node_id": node.node_id,
                        "role": node.role,
                        **node.capacity.as_dict(),
                    }
                    for node in cluster_preview.nodes.values()
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.json(
            {
                "role_compatibility": cluster_preview.role_compatibility,
                "density_limit_semantics_verified": (
                    cluster_preview.density_limit_semantics_verified
                ),
                "assumptions": cluster_preview.assumptions,
            }
        )
    max_workloads = st.number_input(
        "Maximum earliest-arriving workloads",
        min_value=1,
        max_value=100_000,
        value=1_000,
        step=100,
    )
    time_unit = st.text_input(
        "Time-unit label (values are not converted)",
        value="trace_units_unverified",
    )
    selected_policy = st.selectbox(
        "Scheduling policy",
        options=("first_fit", "best_fit", "ifas"),
        format_func=lambda policy: {
            "first_fit": "First Fit",
            "best_fit": "Best Fit",
            "ifas": "Intelligent Fragmentation-Aware (IFAS)",
        }[policy],
    )
    st.caption(
        "First Fit selects the first feasible node by ascending node ID. Best Fit "
        "minimizes normalized residual capacity and breaks ties by node ID. IFAS "
        "scores normalized balance, queued-workload fragmentation, spreading, and residual fit."
    )
    with st.expander("IFAS score configuration", expanded=selected_policy == "ifas"):
        weight_columns = st.columns(4)
        balance_weight = weight_columns[0].slider(
            "Balance weight", 0.0, 2.0, 0.25, step=0.05, key="ifas_balance_weight"
        )
        fragmentation_weight = weight_columns[1].slider(
            "Fragmentation weight", 0.0, 2.0, 0.35, step=0.05, key="ifas_fragmentation_weight"
        )
        spread_weight = weight_columns[2].slider(
            "Spread weight", 0.0, 2.0, 0.20, step=0.05, key="ifas_spread_weight"
        )
        residual_fit_weight = weight_columns[3].slider(
            "Residual-fit weight", 0.0, 2.0, 0.20, step=0.05, key="ifas_residual_fit_weight"
        )
        enable_future_fit = st.checkbox(
            "Enable known-queue future-fit penalty",
            value=False,
            help="Only currently queued workloads are considered; future trace rows are never inspected.",
            key="ifas_enable_future_fit",
        )
        future_fit_weight = st.slider(
            "Future-fit weight",
            0.0,
            2.0,
            0.20,
            step=0.05,
            disabled=not enable_future_fit,
            key="ifas_future_fit_weight",
        )
    try:
        ifas_config = IFASConfig(
            balance_weight=balance_weight,
            fragmentation_weight=fragmentation_weight,
            spread_weight=spread_weight,
            residual_fit_weight=residual_fit_weight,
            enable_future_fit=enable_future_fit,
            future_fit_weight=future_fit_weight,
        )
    except ValueError as exc:
        st.error(f"Invalid IFAS score configuration: {exc}")
        return
    run_column, compare_column = st.columns(2)
    run_selected = run_column.button("Run selected policy")
    compare_both = compare_column.button("Compare First Fit, Best Fit, and IFAS")
    if run_selected or compare_both:
        try:
            frame = load_cleaned_dataset(cleaned_path)
            selected = select_earliest_arrival_rows(frame, int(max_workloads))
            cluster = load_cluster_config(cluster_config_path)
            adapted = adapt_workload_frame(selected, cluster, time_unit=time_unit)
            cluster.assumptions.extend(
                [
                    f"Arrival source: {adapted.arrival_source or 'unavailable; affected rows are excluded'}.",
                    f"Duration source: {adapted.duration_source or 'unavailable; no completion durations are inferred'}.",
                    "Observed lifetimes are applied after simulated allocation; historical scheduling delays are not replayed.",
                ]
            )
            if compare_both:
                st.session_state.pop("gpulytics_simulation", None)
                results = compare_schedulers(
                    cluster,
                    adapted.workloads,
                    rejected_inputs=adapted.rejected,
                    time_unit=time_unit,
                    ifas_config=ifas_config,
                )
                files = save_comparison_report(
                    results,
                    cluster,
                    output_dir / "comparison",
                    input_filename=cleaned_path.name,
                    seed=42,
                    source_row_count=adapted.source_row_count,
                    time_unit=time_unit,
                    ifas_config=ifas_config,
                )
                st.session_state["gpulytics_simulation_comparison"] = {
                    "metrics": {
                        name: comparison_metrics(result)
                        for name, result in results.items()
                    },
                    "report_files": [str(path) for path in files],
                }
            else:
                st.session_state.pop("gpulytics_simulation_comparison", None)
                scheduler = create_scheduler(
                    selected_policy,
                    ifas_config if selected_policy == "ifas" else None,
                )
                result = run_simulation(
                    cluster,
                    adapted.workloads,
                    rejected_inputs=adapted.rejected,
                    policy=scheduler,
                    time_unit=time_unit,
                )
                files = save_simulation_report(
                    result,
                    cluster,
                    output_dir / selected_policy,
                    input_filename=cleaned_path.name,
                    seed=42,
                    source_row_count=adapted.source_row_count,
                )
                st.session_state["gpulytics_simulation"] = {
                    "summary": result.summary,
                    "events": result.events,
                    "files": [str(path) for path in files],
                    "selected_rows": adapted.source_row_count,
                    "rejected_inputs": result.summary["rejected_inputs"],
                }
        except (AnalyticsError, ClusterConfigError, OSError, ValueError) as exc:
            st.error(f"Simulation failed: {exc}")
            return

    saved = st.session_state.get("gpulytics_simulation")
    comparison = st.session_state.get("gpulytics_simulation_comparison")
    if isinstance(comparison, dict):
        st.subheader("Three-scheduler comparison")
        comparison_frame = pd.DataFrame.from_dict(comparison["metrics"], orient="index")
        st.dataframe(comparison_frame, width="stretch")
        st.caption(
            "First Fit, Best Fit, and IFAS use identical ordered workload inputs "
            "and each starts from a fresh copy of the same configured cluster. "
            "Queue delay includes successful placements only; allocation ratios "
            "are not hardware utilization."
        )
        st.subheader("Comparison report")
        for report_file in comparison["report_files"]:
            st.caption(f"`{report_file}`")
    if not isinstance(saved, dict):
        if not isinstance(comparison, dict):
            st.info(
                "Configure the run above and select **Run selected policy** or "
                "**Compare First Fit, Best Fit, and IFAS**."
            )
        return
    st.caption(
        "Results below are from the last explicitly started simulation. Changing "
        "controls does not rerun it; use the action button to generate a new run."
    )
    summary = saved["summary"]
    st.subheader(f"Simulation results — {summary['policy']}")
    metrics = st.columns(4)
    for column, label, key in zip(
        metrics,
        ("Selected input rows", "Allocated", "Completed", "Pending"),
        ("selected_rows", "allocated_workload_count", "completed_workload_count", "pending_workload_count"),
    ):
        value = saved["selected_rows"] if key == "selected_rows" else summary[key]
        column.metric(label, f"{value:,}")
    st.write(
        f"Rejected input rows: **{summary['rejected_workload_count']:,}** · "
        f"Placement attempts: **{summary['placement_attempt_count']:,}** · "
        f"Failed attempts: **{summary['placement_attempt_failure_count']:,}** · "
        f"Queue retries: **{summary['queue_retry_count']:,}**"
    )
    st.caption(summary["metric_definition"])
    st.subheader("Placement failures by reason (unique pending workloads)")
    st.json(summary["placement_failures_by_reason"])
    st.subheader("Time-weighted allocation ratios")
    st.json(summary["time_weighted_allocation_ratios"])
    st.subheader("Time-weighted fragmentation metrics")
    st.json(summary["time_weighted_fragmentation_metrics"])
    delay = summary["queue_delay"]
    st.subheader("Queue delay (successful placements only)")
    st.write(
        f"Observations: {delay['count']:,} · Mean: {delay['mean']} "
        f"{summary['time_unit']} · P95: {delay['p95']} {summary['time_unit']}"
    )
    rejected = saved["rejected_inputs"]
    if rejected:
        st.subheader("Input quality exclusions")
        st.dataframe(rejected, width="stretch", hide_index=True)
    event_frame = pd.DataFrame(saved["events"])
    if not event_frame.empty:
        st.subheader("Event history")
        st.dataframe(event_frame, width="stretch", hide_index=True)
        decisions = event_frame[
            (event_frame["event"] == "allocated")
            & event_frame["decision_diagnostics"].map(bool)
        ]
        if not decisions.empty:
            st.subheader("IFAS placement-score diagnostics")
            st.dataframe(
                decisions[["time", "workload_id", "node_id", "decision_diagnostics"]],
                width="stretch",
                hide_index=True,
            )
    st.subheader("Generated reports and downloads")
    for report_file in saved["files"]:
        path = Path(report_file)
        if path.is_file():
            try:
                st.download_button(
                    f"Download {path.name}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime=(
                        "application/json" if path.suffix == ".json"
                        else "text/csv" if path.suffix == ".csv"
                        else "text/markdown"
                    ),
                    key=f"sim_download_{path.as_posix()}",
                )
            except OSError:
                st.caption(f"`{report_file}`")
        else:
            st.caption(f"`{report_file}`")


def _render_benchmarking() -> None:
    st.header("Scheduler Comparison")
    cleaned_path = PROCESSED_DATA_DIR / "cleaned_workloads.csv"
    cluster_config_path = PROJECT_ROOT / "config" / "demo_cluster.json"
    output_root = REPORT_OUTPUT_DIR / "benchmarks"
    st.warning(
        "Benchmarks compare simulated reservations on the configured hypothetical "
        "cluster. They do not measure actual GPU utilization. Trace time units and "
        "some workload semantics remain unverified."
    )
    if not cleaned_path.is_file():
        st.info(
            "The cleaned workload dataset is unavailable. Run "
            "`python scripts/prepare_dataset.py` from the project directory first."
        )
        return

    st.caption(f"Cleaned workload CSV: `{cleaned_path}`")
    st.caption(f"Cluster configuration: `{cluster_config_path}`")
    st.info(
        "Comparison contract: every selected scheduler receives the same ordered "
        "validated workload inputs and starts from a fresh copy of the same "
        "configured cluster. This page presents the existing evaluation outputs; "
        "it does not run an experiment until you select the explicit button."
    )
    try:
        cluster_preview = load_cluster_config(cluster_config_path)
    except ClusterConfigError as exc:
        st.error(f"Cluster configuration could not be loaded: {exc}")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "node_id": node.node_id,
                    "role": node.role,
                    **node.capacity.as_dict(),
                }
                for node in cluster_preview.nodes.values()
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption("Configured role compatibility and assumptions:")
    st.json(
        {
            "role_compatibility": cluster_preview.role_compatibility,
            "density_limit_semantics_verified": (
                cluster_preview.density_limit_semantics_verified
            ),
            "assumptions": cluster_preview.assumptions,
        }
    )

    selection_method = st.selectbox(
        "Workload selection",
        ("Earliest valid arrivals", "Fixed-seed random sample"),
    )
    subset_limit = st.number_input(
        "Workload count",
        min_value=1,
        max_value=100_000,
        value=1_000,
        step=100,
    )
    seed = st.number_input(
        "Sampling seed",
        min_value=0,
        max_value=2_147_483_647,
        value=42,
        step=1,
    )
    selected_policies = st.multiselect(
        "Scheduling policies",
        options=SUPPORTED_POLICIES,
        default=list(SUPPORTED_POLICIES),
        format_func=lambda policy: {
            "first_fit": "First Fit",
            "best_fit": "Best Fit",
            "ifas": "IFAS",
        }[policy],
    )
    run_sensitivity = st.checkbox(
        "Include one-factor sensitivity scenarios",
        value=False,
        help="Varies one capacity or IFAS-weight setting at a time; scenarios are saved with their configurations.",
    )
    time_unit = st.text_input(
        "Simulation time-unit label (no conversion)",
        value="trace_units_unverified",
    )
    if st.button("Run scheduler benchmark", type="primary"):
        if not selected_policies:
            st.error("Select at least one policy.")
            return
        try:
            source = load_cleaned_dataset(cleaned_path)
            selection_options = (
                {"sample_size": int(subset_limit)}
                if selection_method == "Fixed-seed random sample"
                else {"max_workloads": int(subset_limit)}
            )
            selected, exclusions = select_evaluation_rows(
                source,
                seed=int(seed),
                **selection_options,
            )
            cluster = load_cluster_config(cluster_config_path)
            adapted = adapt_workload_frame(
                selected, cluster, time_unit=time_unit
            )
            if not adapted.workloads:
                st.error(
                    "No valid workloads remain after input validation. Review "
                    "arrival, resource, role, and density-limit exclusions."
                )
                return
            exclusions.extend(
                {
                    "source_row": item.get("source_row"),
                    "workload_id": item.get("workload_id"),
                    "reason": ";".join(item.get("reasons", [])),
                }
                for item in adapted.rejected
            )
            cluster.assumptions.extend(
                [
                    f"Workload arrival source: {adapted.arrival_source or 'missing; affected rows excluded'}.",
                    f"Duration source: {adapted.duration_source or 'unavailable; no completion times inferred'}.",
                    "Historical scheduling delay is not replayed; queue delay is measured within the simulator.",
                ]
            )
            ifas_config = IFASConfig()
            results = run_experiment(
                cluster,
                adapted.workloads,
                policies=selected_policies,
                rejected_inputs=adapted.rejected,
                time_unit=time_unit,
                ifas_config=ifas_config,
            )
            sensitivity_results = (
                run_sensitivity_analysis(
                    cluster,
                    adapted.workloads,
                    policies=selected_policies,
                    rejected_inputs=adapted.rejected,
                    time_unit=time_unit,
                    ifas_config=ifas_config,
                )
                if run_sensitivity
                else []
            )
            experiment_stamp = pd.Timestamp.now(tz="UTC").strftime(
                "%Y%m%dT%H%M%S%fZ"
            )
            experiment_dir = output_root / f"dashboard-{experiment_stamp}"
            files = save_benchmark_outputs(
                results,
                cluster,
                adapted.workloads,
                experiment_dir,
                input_filename=cleaned_path.name,
                source_row_count=len(source),
                selected_row_count=len(selected),
                seed=int(seed),
                time_unit=time_unit,
                ifas_config=ifas_config,
                selection_exclusions=exclusions,
                sensitivity_results=sensitivity_results,
            )
            st.session_state["gpulytics_benchmark"] = {
                "directory": str(experiment_dir),
                "files": [str(path) for path in files],
            }
        except (AnalyticsError, ClusterConfigError, OSError, ValueError, RuntimeError) as exc:
            st.error(f"Benchmark failed: {exc}")
            return

    saved = st.session_state.get("gpulytics_benchmark")
    if not isinstance(saved, dict):
        latest_dir = latest_benchmark_dir(output_root)
        if latest_dir is None:
            st.info(
                "No saved scheduler benchmark is available. Choose policies and "
                "workload selection above, then explicitly run the benchmark."
            )
            return
        saved = {
            "directory": str(latest_dir),
            "files": [
                str(path) for path in list_report_artifacts(latest_dir)
            ],
            "from_saved_report": True,
        }
    benchmark_dir = Path(saved["directory"])
    if saved.get("from_saved_report"):
        st.caption(f"Showing latest saved benchmark report: `{benchmark_dir}`")
    else:
        st.caption(
            "Results below are from the last explicitly started benchmark. "
            "Changing controls does not rerun it."
        )
    comparison_path = benchmark_dir / "scheduler_comparison.csv"
    resource_path = benchmark_dir / "resource_metrics.csv"
    queue_path = benchmark_dir / "queue_delay_metrics.csv"
    fragmentation_path = benchmark_dir / "fragmentation_metrics.csv"
    failure_path = benchmark_dir / "placement_failure_breakdown.csv"
    comparison_artifact = load_report_artifact(comparison_path)
    if comparison_artifact.kind != "csv":
        st.error(
            comparison_artifact.error
            or "The selected benchmark comparison report is unavailable."
        )
        return
    comparison = comparison_artifact.content
    config_artifact = load_report_artifact(
        benchmark_dir / "experiment_config.json"
    )
    if config_artifact.kind == "json" and isinstance(config_artifact.content, dict):
        benchmark_config = config_artifact.content
        recorded_source = benchmark_config.get("source_dataset")
        recorded_rows = benchmark_config.get("source_row_count")
        st.caption(
            f"Saved run input: `{recorded_source or 'not recorded'}`; "
            f"source rows: {recorded_rows if isinstance(recorded_rows, int) else 'not recorded'}. "
            "The benchmark configuration has no source size/mtime fingerprint, so "
            "freshness against the current dataset cannot be confirmed."
        )
    elif config_artifact.kind in {"missing", "error"}:
        st.warning(
            config_artifact.error
            or "Saved benchmark source metadata is unavailable."
        )
    st.subheader("Scheduler comparison")
    st.dataframe(comparison, width="stretch", hide_index=True)
    submitted_counts = (
        pd.to_numeric(comparison["submitted_workloads"], errors="coerce").dropna()
        if "submitted_workloads" in comparison
        else pd.Series(dtype=float)
    )
    submitted = (
        int(submitted_counts.iloc[0])
        if len(submitted_counts) == len(comparison)
        and submitted_counts.nunique() == 1
        else None
    )
    violations = (
        pd.to_numeric(comparison["resource_capacity_violations"], errors="coerce")
        if "resource_capacity_violations" in comparison
        else pd.Series(dtype=float)
    )
    violation_count = (
        int(violations.max())
        if not violations.empty and violations.notna().all()
        else None
    )
    result_columns = st.columns(2)
    result_columns[0].metric(
        "Identical submitted workload count",
        f"{submitted:,}" if submitted is not None else "Unavailable",
    )
    if not violations.empty and violations.notna().all() and violations.eq(0).all():
        result_columns[1].metric("Capacity invariant check", "PASS · 0 violations")
        st.success(
            "The saved evaluation reports zero resource-capacity violations "
            "for every displayed scheduler."
        )
    elif violation_count is not None:
        result_columns[1].metric(
            "Capacity invariant check", f"Review · {violation_count:,}"
        )
        st.error("At least one displayed scheduler reports a capacity violation.")
    else:
        result_columns[1].metric("Capacity invariant check", "Unavailable")
        st.warning(
            "The selected evaluation output does not include a readable capacity "
            "violation field."
        )
    st.caption(
        "Queue delays are simulated arrival-to-first-allocation waits in the "
        "configured time unit. Resource ratios are reserved requests, not measured utilization."
    )
    outcome_columns = [
        column
        for column in (
            "allocated_workloads",
            "completed_workloads",
            "pending_workloads",
            "unique_rejected_workloads",
        )
        if column in comparison.columns
    ]
    if outcome_columns and "policy" in comparison.columns and not comparison.empty:
        st.plotly_chart(
            px.bar(
                comparison.melt(
                    id_vars="policy",
                    value_vars=outcome_columns,
                    var_name="outcome",
                    value_name="workload_count",
                ),
                x="policy",
                y="workload_count",
                color="outcome",
                barmode="group",
                title="Workload outcomes by scheduler",
                labels={"workload_count": "Workload count"},
            ),
            width="stretch",
        )
    queue = _read_report_csv(queue_path)
    queue_columns = [
        column
        for column in (
            "queue_delay_mean",
            "queue_delay_median",
            "queue_delay_p95",
            "queue_delay_maximum",
        )
        if queue is not None and column in queue.columns and queue[column].notna().any()
    ]
    if queue is not None and queue_columns and "policy" in queue.columns and not queue.empty:
        st.plotly_chart(
            px.bar(
                queue.melt(
                    id_vars="policy",
                    value_vars=queue_columns,
                    var_name="statistic",
                    value_name="delay",
                ),
                x="policy",
                y="delay",
                color="statistic",
                barmode="group",
                title="Simulated queue delay",
                labels={"delay": f"Delay ({time_unit})"},
            ),
            width="stretch",
        )
    resources = _read_report_csv(resource_path)
    if resources is not None and "time_weighted_allocation_ratio" in resources.columns:
        available_resources = resources[
            resources["time_weighted_allocation_ratio"].notna()
        ]
    else:
        available_resources = pd.DataFrame()
    if not available_resources.empty and {
        "policy",
        "resource",
        "time_weighted_allocation_ratio",
    }.issubset(available_resources.columns):
        st.plotly_chart(
            px.bar(
                available_resources,
                x="resource",
                y="time_weighted_allocation_ratio",
                color="policy",
                barmode="group",
                title="Time-weighted requested-resource reservations",
                labels={
                    "time_weighted_allocation_ratio": "Reservation ratio (not utilization)"
                },
            ),
            width="stretch",
        )
    fragmentation = _read_report_csv(fragmentation_path)
    if fragmentation is not None and (
        "gpu_fragmentation_blocked_fraction" in fragmentation.columns
        and "snapshot" in fragmentation.columns
        and fragmentation["gpu_fragmentation_blocked_fraction"].notna().any()
    ):
        snapshots = fragmentation[
            (fragmentation["snapshot"] != "time_weighted")
            & fragmentation["gpu_fragmentation_blocked_fraction"].notna()
        ]
        if not snapshots.empty and "policy" in snapshots.columns:
            summary = snapshots.groupby("policy", as_index=False)[
                "gpu_fragmentation_blocked_fraction"
            ].mean()
            st.plotly_chart(
                px.bar(
                    summary,
                    x="policy",
                    y="gpu_fragmentation_blocked_fraction",
                    title="Mean snapshot GPU fragmentation-blocked fraction",
                ),
                width="stretch",
            )
    if not comparison.empty and {"policy", "scheduler_runtime_seconds"}.issubset(comparison.columns):
        st.plotly_chart(
            px.bar(
                comparison,
                x="policy",
                y="scheduler_runtime_seconds",
                title="Scheduler execution time",
                labels={"scheduler_runtime_seconds": "Wall-clock seconds"},
            ),
            width="stretch",
        )
    sensitivity_path = benchmark_dir / "sensitivity" / "sensitivity_results.csv"
    if sensitivity_path.is_file():
        st.subheader("One-factor sensitivity results")
        sensitivity_frame = _read_report_csv(sensitivity_path)
        if sensitivity_frame is not None:
            st.dataframe(sensitivity_frame, width="stretch", hide_index=True)
        if sensitivity_frame is not None and not sensitivity_frame.empty and {"scenario", "allocated_workloads", "policy"}.issubset(sensitivity_frame.columns):
            st.plotly_chart(
                px.line(
                    sensitivity_frame,
                    x="scenario",
                    y="allocated_workloads",
                    color="policy",
                    markers=True,
                    title="Sensitivity of allocations by configuration",
                ),
                width="stretch",
            )
    failures = _read_report_csv(failure_path)
    st.subheader("Placement failure breakdown")
    if failures is None:
        st.info("The saved benchmark has no readable failure-breakdown report.")
    elif failures.empty:
        st.info("No unique pending placement failures were recorded.")
    else:
        st.dataframe(failures, width="stretch", hide_index=True)
    st.caption(f"Full experiment artifacts: `{benchmark_dir}`")
    st.subheader("Download benchmark reports")
    for report_file in saved["files"]:
        path = Path(report_file)
        if path.is_file():
            st.download_button(
                f"Download {path.name}",
                data=path.read_bytes(),
                file_name=path.name,
                mime=(
                    "text/html"
                    if path.suffix == ".html"
                    else "text/csv"
                    if path.suffix == ".csv"
                    else "application/json"
                    if path.suffix == ".json"
                    else "text/markdown"
                ),
                key=f"benchmark_download_{path.as_posix()}",
            )


def _render_ai_predictions() -> None:
    st.header("AI Predictions: Historical Scheduling Delay")
    st.warning(
        "This model estimates historical scheduling delay from workload-creation "
        "features. The timestamp unit is unverified, so predictions are not seconds. "
        "They are not GPU utilization estimates or scheduler-specific performance guarantees."
    )
    cleaned_path = PROCESSED_DATA_DIR / "cleaned_workloads.csv"
    model_path = MODEL_OUTPUT_DIR / "scheduling_delay_pipeline.joblib"
    report_dir = REPORT_OUTPUT_DIR / "ml"
    st.caption(f"Model artifact: `{model_path}`")
    if not cleaned_path.is_file():
        st.info(
            "The cleaned dataset is unavailable. Run "
            "`python scripts/prepare_dataset.py` before training."
        )
        return
    model_status = st.columns(2)
    model_status[0].metric(
        "Training status",
        "Trained model found" if model_path.is_file() else "Not trained",
    )
    model_status[1].metric(
        "Evaluation report",
        "Available" if (report_dir / "evaluation_metrics.json").is_file() else "Not available",
    )
    if st.button("Train / refresh delay model", type="primary"):
        try:
            frame = load_cleaned_dataset(cleaned_path)
            with st.spinner("Training and evaluating chronological candidate models..."):
                trained = train_delay_model(
                    frame,
                    dataset_filename=cleaned_path.name,
                    model_output_dir=MODEL_OUTPUT_DIR,
                    report_output_dir=report_dir,
                    seed=42,
                    test_size=0.2,
                )
            st.session_state["gpulytics_ml_training_result"] = {
                "selected_model": trained["selected_model"],
                "evaluation_metrics": trained["evaluation_metrics"],
                "training_config": trained["training_config"],
            }
            st.success(
                f"Saved {trained['selected_model']} pipeline and held-out reports."
            )
        except (AnalyticsError, TrainingDataError, OSError, ValueError) as exc:
            st.error(f"Model training failed: {exc}")
            return

    config_path = report_dir / "training_config.json"
    metrics_path = report_dir / "evaluation_metrics.json"
    examples_path = report_dir / "prediction_examples.csv"
    feature_path = report_dir / "feature_schema.json"
    if not all(path.is_file() for path in (config_path, metrics_path, feature_path)):
        st.info(
            "No trained model report is available yet. Select **Train / refresh delay model** "
            "to run the explicit training action."
        )
        return
    try:
        evaluation = json.loads(metrics_path.read_text(encoding="utf-8"))
        training = json.loads(config_path.read_text(encoding="utf-8"))
        feature_schema = json.loads(feature_path.read_text(encoding="utf-8"))
        model_bundle = load_delay_model(model_path)
        if (
            not isinstance(evaluation, dict)
            or not isinstance(evaluation.get("test_metrics"), dict)
            or not isinstance(evaluation.get("selected_model"), str)
            or not isinstance(evaluation.get("target_unit"), str)
            or not isinstance(training, dict)
            or not isinstance(training.get("split_row_counts"), dict)
            or not isinstance(feature_schema, dict)
            or not isinstance(feature_schema.get("feature_names"), list)
            or not isinstance(feature_schema.get("forbidden_features"), list)
        ):
            raise ValueError("Saved model reports do not match the expected schema.")
        split_counts = training["split_row_counts"]
        if not {"train", "validation", "test"}.issubset(split_counts):
            raise ValueError("Saved training report is missing split row counts.")
        test_metrics = evaluation["test_metrics"]
        if evaluation["selected_model"] not in test_metrics or any(
            not isinstance(metrics, dict)
            or not {
                "sample_count",
                "mae",
                "median_absolute_error",
                "rmse",
                "r_squared",
            }.issubset(metrics)
            for metrics in test_metrics.values()
        ):
            raise ValueError("Saved evaluation report is missing test metrics.")
    except (
        OSError,
        json.JSONDecodeError,
        PredictionError,
        TypeError,
        ValueError,
    ) as exc:
        st.error(f"Saved model artifacts could not be read: {exc}")
        return

    st.subheader("Chronological held-out evaluation")
    st.caption(
        f"Target: `scheduled_time - creation_time`; unit: {evaluation['target_unit']}. "
        f"Train/validation/test rows: {training['split_row_counts']['train']:,} / "
        f"{training['split_row_counts']['validation']:,} / "
        f"{training['split_row_counts']['test']:,}. "
        f"Selected using validation MAE: **{evaluation['selected_model']}**."
    )
    st.info(
        f"Saved model source: `{training.get('dataset_filename', 'not recorded')}`. "
        "The training configuration does not include a source-file size/mtime "
        "fingerprint, so this saved evaluation cannot be certified as current "
        "against a modified cleaned dataset."
    )
    metric_rows = []
    for model_name, metrics in evaluation["test_metrics"].items():
        metric_rows.append(
            {
                "model": model_name,
                "selected": model_name == evaluation["selected_model"],
                "test_samples": metrics["sample_count"],
                "MAE": metrics["mae"],
                "Median absolute error": metrics["median_absolute_error"],
                "RMSE": metrics["rmse"],
                "R-squared": metrics["r_squared"],
            }
        )
    st.dataframe(pd.DataFrame(metric_rows), width="stretch", hide_index=True)
    st.caption(
        "Test rows were held out from model selection. `median_baseline` and "
        "`dummy_median` are equivalent constant-median references. Errors do not "
        "measure scheduler quality."
    )
    if examples_path.is_file():
        try:
            examples = pd.read_csv(examples_path, low_memory=False)
        except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            st.warning(f"Saved prediction examples could not be read: {exc}")
            examples = pd.DataFrame()
        if len(examples) >= 2 and {"actual_scheduling_delay", "predicted_scheduling_delay"}.issubset(examples.columns):
            st.plotly_chart(
                px.scatter(
                    examples,
                    x="actual_scheduling_delay",
                    y="predicted_scheduling_delay",
                    title="Held-out actual versus predicted historical delay",
                    labels={
                        "actual_scheduling_delay": "Actual delay (source units)",
                        "predicted_scheduling_delay": "Predicted delay (source units)",
                    },
                ),
                width="stretch",
            )
        st.dataframe(examples.head(100), width="stretch", hide_index=True)
    if evaluation.get("test_metrics_by_role"):
        st.subheader("Held-out performance by role (groups with sufficient samples)")
        st.json(evaluation["test_metrics_by_role"].get(evaluation["selected_model"], []))
    st.subheader("Feature schema and model limitations")
    st.write(f"Creation-time features: {', '.join(feature_schema['feature_names'])}")
    st.caption(
        f"Forbidden future/outcome features: {', '.join(feature_schema['forbidden_features'])}"
    )
    for limitation in training.get("limitations", []):
        st.caption(limitation)
    st.caption(
        "Preprocessing is serialized with the estimator. Unknown categories are "
        "ignored safely; omitted optional features are handled by fitted imputers."
    )

    st.subheader("Estimate historical delay for a workload")
    st.caption(
        "Provide the observed creation timestamp in source units and role. Resource "
        "requests are optional input values, not measured consumption."
    )
    try:
        dataset = load_cleaned_dataset(cleaned_path)
        roles = sorted(
            str(value)
            for value in dataset.get("role", pd.Series(dtype=str)).dropna().unique()
        )
    except AnalyticsError:
        roles = []
    if not roles:
        roles = ["HN", "CN"]
    with st.form("delay_prediction_form"):
        creation_time = st.number_input(
            "Creation time (unverified source units)",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )
        role = st.selectbox("Workload role", roles)
        app_name = st.text_input("Application name (optional)", value="")
        resource_columns = st.columns(5)
        requests = {
            "cpu_request": resource_columns[0].number_input(
                "CPU request", min_value=0.0, value=0.0, step=1.0
            ),
            "gpu_request": resource_columns[1].number_input(
                "GPU devices requested", min_value=0, value=0, step=1
            ),
            "memory_request": resource_columns[2].number_input(
                "Memory request", min_value=0.0, value=0.0, step=1.0
            ),
            "rdma_request": resource_columns[3].number_input(
                "RDMA request", min_value=0.0, value=0.0, step=1.0
            ),
            "disk_request": resource_columns[4].number_input(
                "Disk request", min_value=0.0, value=0.0, step=1.0
            ),
        }
        submitted = st.form_submit_button("Predict historical scheduling delay")
    if submitted:
        record = {
            "creation_time": creation_time,
            "role": role,
            "app_name": app_name or None,
            **requests,
        }
        try:
            prediction = predict_scheduling_delay(record, model_bundle)
        except PredictionError as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            if not prediction.empty and "predicted_scheduling_delay" in prediction.columns:
                value = float(prediction.loc[0, "predicted_scheduling_delay"])
                st.metric(
                    "Estimated historical scheduling delay",
                    f"{value:,.3f} source units",
                )
            else:
                st.warning("Prediction result did not contain expected delay values.")
            if not prediction.empty and "interpretation" in prediction.columns:
                st.caption(str(prediction.loc[0, "interpretation"]))
            if not prediction.empty and "input_warnings" in prediction.columns:
                warnings = str(prediction.loc[0, "input_warnings"])
                if warnings:
                    st.warning(warnings)


def _render_algorithm_comparison() -> None:
    st.header("Algorithm Comparison: First Fit, Best Fit & IFAS")
    cleaned_path = PROCESSED_DATA_DIR / "cleaned_workloads.csv"
    cluster_config_path = PROJECT_ROOT / "config" / "demo_cluster.json"
    output_dir = REPORT_OUTPUT_DIR / "simulations" / "comparison"

    st.warning(
        "Controlled, deterministic comparison of First Fit, Best Fit, and "
        "Intelligent Fragmentation-Aware Scheduling (IFAS) on identical ordered "
        "workload inputs. Each policy begins from an identical fresh copy of the "
        "hypothetical cluster. Resource requests are reservations, not measured "
        "hardware utilization. Time units remain in unverified source units."
    )

    if not cleaned_path.is_file():
        st.info(
            "The cleaned workload dataset is unavailable. Run "
            "`python scripts/prepare_dataset.py` from the project directory first."
        )
        return

    st.caption(f"Cleaned dataset: `{cleaned_path}`")
    st.caption(f"Cluster configuration: `{cluster_config_path}`")
    st.caption(f"Comparison reports directory: `{output_dir}`")

    try:
        cluster_preview = load_cluster_config(cluster_config_path)
    except ClusterConfigError as exc:
        st.error(f"Cluster configuration could not be loaded: {exc}")
        return

    with st.expander("Configured hypothetical cluster nodes and constraints"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "node_id": node.node_id,
                        "role": node.role,
                        **node.capacity.as_dict(),
                    }
                    for node in cluster_preview.nodes.values()
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.json(
            {
                "role_compatibility": cluster_preview.role_compatibility,
                "density_limit_semantics_verified": cluster_preview.density_limit_semantics_verified,
                "assumptions": cluster_preview.assumptions,
            }
        )

    col1, col2 = st.columns(2)
    max_workloads = col1.number_input(
        "Maximum earliest-arriving workloads",
        min_value=1,
        max_value=100_000,
        value=1_000,
        step=100,
        key="algo_comp_max_workloads",
    )
    time_unit = col2.text_input(
        "Time-unit label (values are not converted)",
        value="trace_units_unverified",
        key="algo_comp_time_unit",
    )

    with st.expander("IFAS score configuration (used for IFAS run)"):
        wcols = st.columns(4)
        balance_weight = wcols[0].slider(
            "Balance weight", 0.0, 2.0, 0.25, step=0.05, key="algo_comp_ifas_bal"
        )
        fragmentation_weight = wcols[1].slider(
            "Fragmentation weight", 0.0, 2.0, 0.35, step=0.05, key="algo_comp_ifas_frag"
        )
        spread_weight = wcols[2].slider(
            "Spread weight", 0.0, 2.0, 0.20, step=0.05, key="algo_comp_ifas_spread"
        )
        residual_fit_weight = wcols[3].slider(
            "Residual-fit weight", 0.0, 2.0, 0.20, step=0.05, key="algo_comp_ifas_res"
        )
        enable_future_fit = st.checkbox(
            "Enable known-queue future-fit penalty",
            value=False,
            help="Only currently queued workloads are considered; future trace rows are never inspected.",
            key="algo_comp_ifas_future",
        )
        future_fit_weight = st.slider(
            "Future-fit weight",
            0.0,
            2.0,
            0.20,
            step=0.05,
            disabled=not enable_future_fit,
            key="algo_comp_ifas_future_w",
        )

    try:
        ifas_config = IFASConfig(
            balance_weight=balance_weight,
            fragmentation_weight=fragmentation_weight,
            spread_weight=spread_weight,
            residual_fit_weight=residual_fit_weight,
            enable_future_fit=enable_future_fit,
            future_fit_weight=future_fit_weight,
        )
    except ValueError as exc:
        st.error(f"Invalid IFAS score configuration: {exc}")
        return

    if st.button("Run algorithm comparison", type="primary", key="algo_comp_run_btn"):
        try:
            frame = load_cleaned_dataset(cleaned_path)
            selected = select_earliest_arrival_rows(frame, int(max_workloads))
            cluster = load_cluster_config(cluster_config_path)
            adapted = adapt_workload_frame(selected, cluster, time_unit=time_unit)
            cluster.assumptions.extend(
                [
                    f"Arrival source: {adapted.arrival_source or 'unavailable; affected rows are excluded'}.",
                    f"Duration source: {adapted.duration_source or 'unavailable; no completion durations are inferred'}.",
                    "Observed lifetimes are applied after simulated allocation; historical scheduling delays are not replayed.",
                ]
            )
            with st.spinner("Executing First Fit, Best Fit, and IFAS across fresh cluster states..."):
                results = compare_schedulers(
                    cluster,
                    adapted.workloads,
                    rejected_inputs=adapted.rejected,
                    time_unit=time_unit,
                    ifas_config=ifas_config,
                )
                files = save_comparison_report(
                    results,
                    cluster,
                    output_dir,
                    input_filename=cleaned_path.name,
                    seed=42,
                    source_row_count=adapted.source_row_count,
                    time_unit=time_unit,
                    ifas_config=ifas_config,
                )
            st.session_state["gpulytics_algorithm_comparison"] = {
                "metrics": {
                    name: comparison_metrics(result)
                    for name, result in results.items()
                },
                "report_files": [str(path) for path in files],
                "from_saved_report": False,
            }
            st.success("Algorithm comparison finished successfully.")
        except (AnalyticsError, ClusterConfigError, OSError, ValueError) as exc:
            st.error(f"Algorithm comparison failed: {exc}")
            return

    saved = st.session_state.get("gpulytics_algorithm_comparison")
    if not isinstance(saved, dict):
        summary_path = output_dir / "comparison_summary.json"
        metrics_csv_path = output_dir / "comparison_metrics.csv"
        if metrics_csv_path.is_file():
            try:
                metrics_df = pd.read_csv(metrics_csv_path)
                if not metrics_df.empty:
                    saved = {
                        "dataframe": metrics_df,
                        "report_files": [
                            str(p)
                            for p in (summary_path, metrics_csv_path, output_dir / "comparison_report.md")
                            if p.is_file()
                        ],
                        "from_saved_report": True,
                    }
            except Exception:
                saved = None

    if not isinstance(saved, dict):
        st.info(
            "No algorithm comparison results yet. Configure settings above and "
            "select **Run algorithm comparison**."
        )
        return

    if saved.get("from_saved_report"):
        st.caption(f"Showing saved comparison results from `{output_dir}`.")
    else:
        st.caption("Results below are from the latest explicitly executed run.")

    if "dataframe" in saved:
        comparison_df = saved["dataframe"]
    elif "metrics" in saved:
        comparison_df = pd.DataFrame.from_dict(saved["metrics"], orient="index")
    else:
        comparison_df = pd.DataFrame()

    if comparison_df.empty:
        st.info("No metric data found in comparison result.")
        return
    if "policy" not in comparison_df.columns:
        comparison_df["policy"] = comparison_df.index

    # Summary Metric Cards
    mcols = st.columns(3)
    submitted_val = None
    if "submitted_workloads" in comparison_df.columns:
        sub_series = pd.to_numeric(comparison_df["submitted_workloads"], errors="coerce").dropna()
        if not sub_series.empty:
            submitted_val = int(sub_series.iloc[0])
    mcols[0].metric(
        "Submitted Workloads",
        f"{submitted_val:,}" if submitted_val is not None else "N/A",
    )

    allocated_val = None
    if "allocated_workloads" in comparison_df.columns:
        alloc_series = pd.to_numeric(comparison_df["allocated_workloads"], errors="coerce").dropna()
        if not alloc_series.empty:
            allocated_val = int(alloc_series.iloc[0])
    mcols[1].metric(
        "Allocated Workloads",
        f"{allocated_val:,}" if allocated_val is not None else "N/A",
    )

    violations_series = (
        pd.to_numeric(comparison_df["resource_capacity_violations"], errors="coerce")
        if "resource_capacity_violations" in comparison_df.columns
        else pd.Series(dtype=float)
    )
    if not violations_series.empty and violations_series.eq(0).all():
        mcols[2].metric("Capacity Invariant", "PASS · 0 violations")
        st.success("All algorithms adhered strictly to resource capacities with 0 violations.")
    elif not violations_series.empty and violations_series.gt(0).any():
        max_viol = int(violations_series.max())
        mcols[2].metric("Capacity Invariant", f"FAIL · {max_viol} violations")
        st.error(f"Capacity violations detected: {max_viol}")
    else:
        mcols[2].metric("Capacity Invariant", "Unavailable")

    st.subheader("Side-by-side metric comparison")
    st.dataframe(comparison_df, width="stretch")

    st.info(
        "Genuine algorithm comparison: All three schedulers receive identical "
        "ordered workloads and run against fresh copies of the same cluster. Under "
        "these trace constraints and hypothetical cluster limits, placement outcomes "
        "are driven by capacity bottlenecks. IFAS records multi-resource balance and "
        "fragmentation diagnostics for each decision without fabricating superior results."
    )

    # Comparative charts
    chart_col1, chart_col2 = st.columns(2)
    outcome_cols = [c for c in ("allocated_workloads", "completed_workloads", "pending_workloads") if c in comparison_df.columns]
    if outcome_cols and "policy" in comparison_df.columns and not comparison_df.empty:
        melted_outcomes = comparison_df.melt(
            id_vars="policy",
            value_vars=outcome_cols,
            var_name="outcome",
            value_name="count",
        )
        chart_col1.plotly_chart(
            px.bar(
                melted_outcomes,
                x="policy",
                y="count",
                color="outcome",
                barmode="group",
                title="Workload placement outcomes by algorithm",
                labels={"count": "Workloads", "policy": "Algorithm"},
            ),
            width="stretch",
        )

    delay_cols = [c for c in ("average_queue_delay", "p95_queue_delay") if c in comparison_df.columns]
    if delay_cols and "policy" in comparison_df.columns and not comparison_df.empty:
        melted_delay = comparison_df.melt(
            id_vars="policy",
            value_vars=delay_cols,
            var_name="metric",
            value_name="delay",
        )
        chart_col2.plotly_chart(
            px.bar(
                melted_delay,
                x="policy",
                y="delay",
                color="metric",
                barmode="group",
                title="Simulated queue delay by algorithm",
                labels={"delay": f"Delay ({time_unit})", "policy": "Algorithm"},
            ),
            width="stretch",
        )

    # Runtime comparison
    if "scheduler_runtime_seconds" in comparison_df.columns and "policy" in comparison_df.columns and not comparison_df.empty:
        st.plotly_chart(
            px.bar(
                comparison_df,
                x="policy",
                y="scheduler_runtime_seconds",
                title="Scheduler execution time (wall-clock seconds)",
                labels={"scheduler_runtime_seconds": "Wall-clock seconds", "policy": "Algorithm"},
                color="policy",
            ),
            width="stretch",
        )

    # Downloads
    st.subheader("Comparison report artifacts")
    for report_file in saved.get("report_files", []):
        path = Path(report_file)
        if path.is_file():
            try:
                st.download_button(
                    f"Download {path.name}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime=(
                        "text/csv" if path.suffix == ".csv"
                        else "application/json" if path.suffix == ".json"
                        else "text/markdown"
                    ),
                    key=f"algo_comp_dl_{path.as_posix()}",
                )
            except OSError:
                st.caption(f"`{report_file}`")
        else:
            st.caption(f"`{report_file}`")


if __name__ == "__main__":
    main()
