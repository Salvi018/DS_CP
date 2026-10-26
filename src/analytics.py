"""Reusable descriptive analytics for prepared GPU workload traces.

Resource fields describe requested resources, not measured consumption. All
timestamp-derived values retain their source units until those units are
verified by authoritative dataset documentation.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import PROCESSED_DATA_DIR, REPORT_OUTPUT_DIR
from src.dataset_inspector import RESOURCE_NAMES

LOGGER = logging.getLogger(__name__)

REQUEST_COLUMNS = tuple(f"{resource}_request" for resource in RESOURCE_NAMES)
LIMIT_COLUMNS = tuple(f"{resource}_limit" for resource in RESOURCE_NAMES)
TIMING_COLUMNS = ("scheduling_delay_raw_units", "observed_lifetime_raw_units")
DEFAULT_CLASSIFICATION_THRESHOLDS = {
    "small": {"cpu_request": 8.0, "gpu_request": 0.0, "memory_request": 64.0},
    "large": {"cpu_request": 32.0, "gpu_request": 2.0, "memory_request": 256.0},
}


class AnalyticsError(Exception):
    """Raised when a workload dataset cannot be analyzed."""


@dataclass
class AnalyticsResult:
    """Computed summaries and an analysis-only classified data frame."""

    overview: dict[str, Any]
    resource_summary: pd.DataFrame
    role_summary: pd.DataFrame
    application_summary: pd.DataFrame
    scheduling_delay_summary: dict[str, Any] | None
    lifetime_summary: dict[str, Any] | None
    correlation_matrix: pd.DataFrame
    gpu_comparison: pd.DataFrame
    workload_size_summary: pd.DataFrame
    missing_summary: pd.DataFrame
    data_quality_summary: pd.DataFrame
    classified_workloads: pd.DataFrame
    unavailable_analyses: dict[str, str]


def load_cleaned_dataset(path: str | Path = PROCESSED_DATA_DIR / "cleaned_workloads.csv") -> pd.DataFrame:
    """Load the prepared CSV, reporting missing or malformed input clearly."""
    source = Path(path)
    if not source.is_file():
        raise AnalyticsError(
            f"Cleaned workload dataset not found at {source}. "
            "Run `python scripts/prepare_dataset.py` first."
        )
    try:
        frame = pd.read_csv(source, low_memory=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise AnalyticsError(f"Could not read cleaned workload dataset {source}: {exc}") from exc
    if frame.empty and len(frame.columns) == 0:
        raise AnalyticsError(f"Cleaned workload dataset has no header: {source}")
    return frame


def _numeric_series(frame: pd.DataFrame, column: str) -> tuple[pd.Series, int]:
    original = frame[column]
    numeric = pd.to_numeric(original, errors="coerce")
    invalid = int((original.notna() & numeric.isna()).sum())
    non_finite = numeric.notna() & ~np.isfinite(numeric)
    invalid += int(non_finite.sum())
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric, invalid


def numeric_summary(frame: pd.DataFrame, column: str) -> dict[str, Any] | None:
    """Summarize finite numeric observations and report missing/invalid values."""
    if column not in frame:
        return None
    values, invalid_count = _numeric_series(frame, column)
    observed = values.dropna()
    return {
        "column": column,
        "count": int(observed.count()),
        "mean": float(observed.mean()) if len(observed) else None,
        "median": float(observed.median()) if len(observed) else None,
        "std": float(observed.std(ddof=1)) if len(observed) > 1 else None,
        "min": float(observed.min()) if len(observed) else None,
        "p25": float(observed.quantile(0.25)) if len(observed) else None,
        "p75": float(observed.quantile(0.75)) if len(observed) else None,
        "max": float(observed.max()) if len(observed) else None,
        "missing_count": int(values.isna().sum()),
        "invalid_count": invalid_count,
    }


def _timing_summary(frame: pd.DataFrame, column: str) -> dict[str, Any] | None:
    if column not in frame:
        return None
    values, invalid_count = _numeric_series(frame, column)
    valid = values.dropna()
    negative = valid[valid < 0]
    usable = valid[valid >= 0]
    return {
        "column": column,
        "unit": "unverified source timestamp units; values are not seconds",
        "valid_observation_count": int(valid.count()),
        "usable_non_negative_count": int(usable.count()),
        "missing_count": int(values.isna().sum()),
        "invalid_count": invalid_count,
        "negative_count": int(negative.count()),
        "negative_min": float(negative.min()) if len(negative) else None,
        "negative_max": float(negative.max()) if len(negative) else None,
        "mean": float(usable.mean()) if len(usable) else None,
        "median": float(usable.median()) if len(usable) else None,
        "p95": float(usable.quantile(0.95)) if len(usable) else None,
        "p99": float(usable.quantile(0.99)) if len(usable) else None,
        "min": float(usable.min()) if len(usable) else None,
        "max": float(usable.max()) if len(usable) else None,
    }


def classify_workload_size(
    frame: pd.DataFrame,
    thresholds: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Classify workloads with transparent analytical request thresholds.

    A row is small when every available CPU/GPU/memory request is at or below
    its small threshold, large when any observed request is at or above its
    large threshold, and medium otherwise. Missing fields are ignored; rows
    with no available classification resources are ``unclassified``.
    """
    limits = thresholds or DEFAULT_CLASSIFICATION_THRESHOLDS
    if set(limits) != {"small", "large"}:
        raise ValueError("thresholds must define exactly 'small' and 'large'")
    small_limits, large_limits = limits["small"], limits["large"]
    if set(small_limits) != set(large_limits):
        raise ValueError("small and large thresholds must define the same resource fields")
    for column, small_limit in small_limits.items():
        large_limit = large_limits[column]
        if (
            not math.isfinite(float(small_limit))
            or not math.isfinite(float(large_limit))
            or float(small_limit) > float(large_limit)
        ):
            raise ValueError(f"Invalid classification thresholds for {column}")
    eligible = [
        column
        for column in ("cpu_request", "gpu_request", "memory_request")
        if column in frame and column in small_limits and column in large_limits
    ]
    result = frame.copy()
    if not eligible:
        result["workload_size"] = "unclassified"
        result["workload_size_resource_count"] = 0
        result["workload_size_basis"] = ""
        return result

    if frame.empty:
        result["workload_size"] = pd.Series(dtype="string", index=frame.index)
        result["workload_size_resource_count"] = pd.Series(dtype="int64", index=frame.index)
        result["workload_size_basis"] = pd.Series(dtype="string", index=frame.index)
        return result

    numeric = pd.DataFrame(
        {column: pd.to_numeric(frame[column], errors="coerce") for column in eligible},
        index=frame.index,
    )
    observed_count = numeric.notna().sum(axis=1)
    small = pd.Series(True, index=frame.index)
    large = pd.Series(False, index=frame.index)
    for column in eligible:
        values = numeric[column]
        small &= values.isna() | (values <= small_limits[column])
        large |= values.notna() & (values >= large_limits[column])
    labels = pd.Series("medium", index=frame.index, dtype="string")
    labels.loc[small & observed_count.gt(0)] = "small"
    labels.loc[large] = "large"
    labels.loc[observed_count.eq(0)] = "unclassified"
    result["workload_size"] = labels
    result["workload_size_resource_count"] = observed_count.astype("int64")
    result["workload_size_basis"] = numeric.notna().apply(
        lambda row: ",".join(row.index[row].tolist()), axis=1
    )
    return result


def _category_summary(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in frame:
        return pd.DataFrame(columns=[column, "workload_count", "missing_count"])
    counts = frame[column].value_counts(dropna=True).rename("workload_count").reset_index()
    counts = counts.rename(columns={"index": column, frame[column].name: column})
    if column not in counts:
        counts = counts.rename(columns={counts.columns[0]: column})
    counts["missing_count"] = int(frame[column].isna().sum())
    return counts


def _application_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if "app_name" not in frame:
        return pd.DataFrame(columns=["app_name", "workload_count"])
    aggregations: dict[str, tuple[str, str]] = {"workload_count": ("app_name", "size")}
    for resource in RESOURCE_NAMES:
        column = f"{resource}_request"
        if column in frame:
            aggregations[f"{resource}_requested_total"] = (column, lambda s: s.sum(min_count=1))
            aggregations[f"{resource}_requested_mean"] = (column, "mean")
    result = frame.groupby("app_name", dropna=False).agg(**aggregations).reset_index()
    return result.sort_values(["workload_count", "app_name"], ascending=[False, True]).reset_index(
        drop=True
    )


def _resource_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in (*REQUEST_COLUMNS, *LIMIT_COLUMNS):
        summary = numeric_summary(frame, column)
        if summary is not None:
            summary["metric_kind"] = "requested" if column.endswith("_request") else "limit"
            rows.append(summary)
    return pd.DataFrame(rows)


def _correlations(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in REQUEST_COLUMNS if column in frame]
    numeric = pd.DataFrame(
        {column: _numeric_series(frame, column)[0] for column in columns},
        index=frame.index,
    )
    if len(columns) < 2:
        return pd.DataFrame()
    return numeric.corr(min_periods=2)


def _gpu_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    if "gpu_request" not in frame:
        return pd.DataFrame()
    gpu_values, _ = _numeric_series(frame, "gpu_request")
    groups = pd.Series(index=frame.index, dtype="string")
    groups.loc[gpu_values.gt(0)] = "gpu_requesting"
    groups.loc[gpu_values.eq(0)] = "cpu_only_zero_gpu_request"
    records = []
    for group in ("gpu_requesting", "cpu_only_zero_gpu_request"):
        mask = groups.eq(group)
        if not mask.any():
            continue
        record: dict[str, Any] = {"workload_group": group, "workload_count": int(mask.sum())}
        for column in REQUEST_COLUMNS:
            if column not in frame:
                continue
            values, invalid = _numeric_series(frame.loc[mask], column)
            record[f"{column}_mean"] = float(values.mean()) if values.notna().any() else None
            record[f"{column}_median"] = float(values.median()) if values.notna().any() else None
            record[f"{column}_missing_count"] = int(values.isna().sum())
            record[f"{column}_invalid_count"] = invalid
        records.append(record)
    return pd.DataFrame(records)


def _quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if "validation_issue_count" in frame:
        values, _ = _numeric_series(frame, "validation_issue_count")
        rows.extend(
            [
                {
                    "finding": "rows_with_validation_issues",
                    "count": int(values.gt(0).sum()),
                    "details": "Rows with one or more flags recorded by preprocessing.",
                },
                {
                    "finding": "validation_issue_occurrences",
                    "count": int(values.sum()),
                    "details": "Sum of per-row issue counts; not a unique-row count.",
                },
            ]
        )
    if "data_quality_flags" in frame:
        flags = frame["data_quality_flags"].fillna("").astype(str)
        categories: dict[str, int] = {}
        for entry in flags:
            for category in filter(None, entry.split(";")):
                categories[category] = categories.get(category, 0) + 1
        rows.extend(
            {
                "finding": f"flag:{category}",
                "count": count,
                "details": "Rows carrying this preprocessing data-quality flag.",
            }
            for category, count in sorted(categories.items())
        )
    for column in frame.columns:
        if column.endswith("_raw"):
            rows.append(
                {
                    "finding": f"invalid_raw_values:{column}",
                    "count": int(frame[column].notna().sum()),
                    "details": "Original invalid/non-finite values preserved by preprocessing.",
                }
            )
    return pd.DataFrame(rows, columns=["finding", "count", "details"])


def analyze_workloads(
    frame: pd.DataFrame,
    *,
    dataset_filename: str = "<dataframe>",
    thresholds: dict[str, dict[str, float]] | None = None,
) -> AnalyticsResult:
    """Compute schema-aware descriptive summaries without mutating input."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty and len(frame.columns) == 0:
        raise AnalyticsError("Workload dataset has no columns.")
    classified = classify_workload_size(frame, thresholds)
    unavailable: dict[str, str] = {}
    for name, required in {
        "role_summary": ("role",),
        "application_summary": ("app_name",),
        "resource_summary": REQUEST_COLUMNS,
        "scheduling_delay": ("scheduling_delay_raw_units",),
        "observed_lifetime": ("observed_lifetime_raw_units",),
        "correlations": REQUEST_COLUMNS,
        "gpu_comparison": ("gpu_request",),
    }.items():
        missing = [column for column in required if column not in frame]
        if missing:
            unavailable[name] = "Required columns unavailable: " + ", ".join(missing)

    resource_summary = _resource_summary(frame)
    role_summary = _category_summary(frame, "role")
    application_summary = _application_summary(frame)
    delay_summary = _timing_summary(frame, "scheduling_delay_raw_units")
    lifetime_summary = _timing_summary(frame, "observed_lifetime_raw_units")
    correlations = _correlations(frame)
    gpu_comparison = _gpu_comparison(frame)
    missing = pd.DataFrame(
        [
            {
                "column": column,
                "missing_count": int(frame[column].isna().sum()),
                "missing_percent": float(frame[column].isna().mean() * 100),
            }
            for column in frame.columns
        ]
    )
    quality = _quality_summary(frame)
    workload_sizes = (
        classified["workload_size"]
        .value_counts(dropna=False)
        .rename_axis("workload_size")
        .rename("workload_count")
        .reset_index()
    )
    gpu_values = (
        _numeric_series(frame, "gpu_request")[0]
        if "gpu_request" in frame
        else pd.Series(dtype=float)
    )
    overview: dict[str, Any] = {
        "dataset_filename": dataset_filename,
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": [{"name": str(column), "dtype": str(dtype)} for column, dtype in frame.dtypes.items()],
        "application_group_count": (
            int(frame["app_name"].nunique(dropna=True)) if "app_name" in frame else None
        ),
        "gpu_requesting_workload_count": (
            int(gpu_values.gt(0).sum()) if "gpu_request" in frame else None
        ),
        "cpu_only_workload_count": (
            int(gpu_values.eq(0).sum()) if "gpu_request" in frame else None
        ),
        "workload_size_thresholds": thresholds or DEFAULT_CLASSIFICATION_THRESHOLDS,
        "workload_size_threshold_note": (
            "Configurable analytical heuristics, not official Alibaba workload labels. "
            "Available CPU/GPU/memory request fields are considered; missing fields are "
            "ignored, and rows with no available classification resource are unclassified."
        ),
        "resource_metric_note": (
            "Resource columns represent requests and configured limits, not measured consumption, "
            "GPU utilization, temperature, or power."
        ),
        "timestamp_unit_assumption": (
            "Unverified source timestamp units. Scheduling delay and lifetime are not converted to seconds."
        ),
        "observed_lifetime_limitation": (
            "Observed scheduled lifetime is not necessarily the full workload lifetime when observation is incomplete."
        ),
        "unavailable_analyses": unavailable,
        "definitions": {
            "resource summary": "Finite numeric values in each observed request/limit column; missing and invalid counts are explicit.",
            "scheduling delay": "scheduling_delay_raw_units, excluding negative values from distribution statistics and reporting them separately.",
            "observed lifetime": "observed_lifetime_raw_units, excluding negative values from distribution statistics and reporting them separately.",
            "GPU-requesting workloads": "Rows with a numeric gpu_request greater than zero.",
            "CPU-only workloads": "Rows with a numeric gpu_request equal to zero; this does not assert absence of every other accelerator.",
            "correlation": "Pairwise Pearson correlation across resource-request columns; descriptive association, not causation.",
        },
    }
    return AnalyticsResult(
        overview=overview,
        resource_summary=resource_summary,
        role_summary=role_summary,
        application_summary=application_summary,
        scheduling_delay_summary=delay_summary,
        lifetime_summary=lifetime_summary,
        correlation_matrix=correlations,
        gpu_comparison=gpu_comparison,
        workload_size_summary=workload_sizes,
        missing_summary=missing,
        data_quality_summary=quality,
        classified_workloads=classified,
        unavailable_analyses=unavailable,
    )


def plot_resource_distributions(frame: pd.DataFrame) -> go.Figure | None:
    """Create request distributions for available CPU, GPU, and memory fields."""
    columns = [column for column in ("cpu_request", "gpu_request", "memory_request") if column in frame]
    if not columns:
        return None
    figure = make_subplots(rows=1, cols=len(columns), subplot_titles=columns)
    for position, column in enumerate(columns, start=1):
        values, _ = _numeric_series(frame, column)
        values = values.dropna()
        if len(values):
            skewed = bool(values.ge(0).all() and values.skew() > 1.5)
            plotted_values = np.log1p(values) if skewed else values
            figure.add_trace(
                go.Histogram(x=plotted_values, name=column, nbinsx=40, showlegend=False),
                row=1,
                col=position,
            )
        axis_label = f"{column} (requested)"
        if len(values) and values.ge(0).all() and values.skew() > 1.5:
            axis_label += " — log1p scale"
        figure.update_xaxes(title_text=axis_label, row=1, col=position)
        figure.update_yaxes(title_text="Workload count", row=1, col=position)
    figure.update_layout(title="Resource request distributions", height=400)
    return figure


def plot_role_distribution(frame: pd.DataFrame) -> go.Figure | None:
    summary = _category_summary(frame, "role")
    if summary.empty:
        return None
    return px.bar(summary, x="role", y="workload_count", title="Workload counts by role")


def plot_top_applications(frame: pd.DataFrame, top_n: int = 20) -> go.Figure | None:
    if top_n < 1:
        raise ValueError("top_n must be positive")
    summary = _application_summary(frame)
    if summary.empty:
        return None
    top = summary.head(top_n)
    return px.bar(
        top.sort_values("workload_count"),
        x="workload_count",
        y="app_name",
        orientation="h",
        title=f"Top {min(top_n, len(top))} applications by workload count",
    )


def _plot_timing_distribution(frame: pd.DataFrame, column: str, title: str) -> go.Figure | None:
    if column not in frame:
        return None
    values, _ = _numeric_series(frame, column)
    usable = values[values.ge(0)].dropna()
    if usable.empty:
        return None
    return px.histogram(
        x=usable,
        nbins=50,
        title=title,
        labels={"x": f"{column} (unverified source units)", "y": "Workload count"},
    )


def plot_scheduling_delay_distribution(frame: pd.DataFrame) -> go.Figure | None:
    return _plot_timing_distribution(
        frame, "scheduling_delay_raw_units", "Scheduling delay distribution (non-negative observations)"
    )


def plot_lifetime_distribution(frame: pd.DataFrame) -> go.Figure | None:
    return _plot_timing_distribution(
        frame, "observed_lifetime_raw_units", "Observed scheduled lifetime (non-negative observations)"
    )


def plot_resource_correlation(frame: pd.DataFrame) -> go.Figure | None:
    correlations = _correlations(frame)
    if correlations.empty or correlations.isna().all().all():
        return None
    return px.imshow(
        correlations,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu",
        title="Resource request correlations (association, not causation)",
    )


def plot_request_vs_limit(frame: pd.DataFrame) -> go.Figure | None:
    resources = [
        resource
        for resource in RESOURCE_NAMES
        if f"{resource}_request" in frame and f"{resource}_limit" in frame
    ]
    if not resources:
        return None
    figure = make_subplots(rows=1, cols=len(resources), subplot_titles=resources)
    for position, resource in enumerate(resources, start=1):
        request, _ = _numeric_series(frame, f"{resource}_request")
        limit, _ = _numeric_series(frame, f"{resource}_limit")
        valid = request.notna() & limit.notna()
        figure.add_trace(
            go.Scatter(
                x=request[valid],
                y=limit[valid],
                mode="markers",
                marker={"size": 5, "opacity": 0.5},
                name=resource,
                showlegend=False,
            ),
            row=1,
            col=position,
        )
        figure.update_xaxes(title_text=f"{resource} requested", row=1, col=position)
        figure.update_yaxes(title_text=f"{resource} limit", row=1, col=position)
    figure.update_layout(title="Requested resources versus configured limits", height=450)
    return figure


def create_analytics_figures(frame: pd.DataFrame) -> dict[str, go.Figure]:
    """Build only charts supported by the observed dataset columns and values."""
    builders = {
        "resource_distributions": lambda: plot_resource_distributions(frame),
        "role_distribution": lambda: plot_role_distribution(frame),
        "top_applications": lambda: plot_top_applications(frame),
        "scheduling_delay_distribution": lambda: plot_scheduling_delay_distribution(frame),
        "lifetime_distribution": lambda: plot_lifetime_distribution(frame),
        "resource_correlation": lambda: plot_resource_correlation(frame),
        "resource_request_vs_limit": lambda: plot_request_vs_limit(frame),
    }
    figures: dict[str, go.Figure] = {}
    for name, build in builders.items():
        figure = build()
        if figure is not None:
            figures[name] = figure
    return figures


def _json_clean(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _markdown_report(result: AnalyticsResult) -> str:
    overview = result.overview
    resources = result.resource_summary
    roles = result.role_summary
    applications = result.application_summary.head(20)
    delay = result.scheduling_delay_summary
    lifetime = result.lifetime_summary
    quality = result.data_quality_summary
    chart_references = result.overview.get("generated_charts", [])

    def table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_Unavailable: required columns or valid observations are not present._"
        displayed = frame.fillna("")
        headers = [str(column) for column in displayed.columns]
        rows = [
            [
                str(value).replace("|", "\\|").replace("\n", " ")
                for value in row
            ]
            for row in displayed.itertuples(index=False, name=None)
        ]
        separator = ["---"] * len(headers)
        return "\n".join(
            ["| " + " | ".join(headers) + " |", "| " + " | ".join(separator) + " |"]
            + ["| " + " | ".join(row) + " |" for row in rows]
        )

    return f"""# GPUlytics workload analytics

- **Dataset:** `{overview['dataset_filename']}`
- **Analysis timestamp (UTC):** {overview['analysis_timestamp_utc']}
- **Workloads:** {overview['row_count']:,}
- **Columns:** {overview['column_count']}
- **Application groups:** {overview['application_group_count']}
- **GPU-requesting workloads:** {overview['gpu_requesting_workload_count']}
- **Timestamp units:** {overview['timestamp_unit_assumption']}

## Resource request and limit summaries

Resource fields are requests/configured limits, not measured consumption or utilization.

{table(resources)}

## Role distribution

{table(roles)}

## Top applications

Top applications by workload count, with requested-resource aggregates where columns are available:

{table(applications)}

The full application table is saved in `application_summary.csv`.

## Scheduling delay

```json
{json.dumps(_json_clean(delay), indent=2)}
```

Statistics use non-negative, finite observations. Negative, missing, and invalid observations are reported separately. Units remain unverified and are not seconds.

## Observed scheduled lifetime

```json
{json.dumps(_json_clean(lifetime), indent=2)}
```

An observed lifetime is not necessarily the true full workload lifetime when the observation window is incomplete.

## GPU-requesting versus CPU-only workloads

{table(result.gpu_comparison)}

## Workload size heuristic

{table(result.workload_size_summary)}

Thresholds: `{json.dumps(overview['workload_size_thresholds'], sort_keys=True)}`. These are configurable analytical heuristics, not official Alibaba labels. Missing resource fields are ignored; no available resource values yield `unclassified`.

## Missingness and data quality

{table(result.missing_summary.sort_values("missing_percent", ascending=False))}

{table(quality)}

## Correlations

{table(result.correlation_matrix.reset_index(names="resource_request"))}

Pairwise Pearson correlations describe association and do not establish causality.

## Interactive charts

{chr(10).join(f'- [{Path(reference).stem.replace("_", " ").title()}]({reference})' for reference in chart_references) or '_No charts were available for the observed columns._'}

## Limitations and metric definitions

{json.dumps(overview['definitions'], indent=2)}

- Scheduling delay is calculated from the preprocessed historical delay field.
- GPU-requesting means a numeric GPU request greater than zero; zero-request workloads remain a separate valid group.
- Requested-resource features do not measure actual GPU utilization, power, temperature, or resource consumption.
- Negative and missing time/resource values are excluded only from the relevant numeric statistic or plot and are represented in the quality summaries.
- Unavailable analyses are recorded in `overview.json`.
"""


def generate_analytics(
    frame: pd.DataFrame,
    output_dir: str | Path = REPORT_OUTPUT_DIR / "analytics",
    *,
    dataset_filename: str = "<dataframe>",
) -> AnalyticsResult:
    """Generate deterministic tabular/JSON/Markdown analytics outputs and Plotly HTML."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = analyze_workloads(frame, dataset_filename=dataset_filename)
    (output / "overview.json").write_text(
        json.dumps(_json_clean(result.overview), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    result.resource_summary.to_csv(output / "resource_summary.csv", index=False)
    if not result.role_summary.empty:
        result.role_summary.to_csv(output / "role_summary.csv", index=False)
    else:
        (output / "role_summary.csv").unlink(missing_ok=True)
    if not result.application_summary.empty:
        result.application_summary.to_csv(output / "application_summary.csv", index=False)
    else:
        (output / "application_summary.csv").unlink(missing_ok=True)
    if result.scheduling_delay_summary is not None:
        (output / "scheduling_delay_summary.json").write_text(
            json.dumps(_json_clean(result.scheduling_delay_summary), indent=2, allow_nan=False),
            encoding="utf-8",
        )
    else:
        (output / "scheduling_delay_summary.json").unlink(missing_ok=True)
    if result.lifetime_summary is not None:
        (output / "lifetime_summary.json").write_text(
            json.dumps(_json_clean(result.lifetime_summary), indent=2, allow_nan=False),
            encoding="utf-8",
        )
    else:
        (output / "lifetime_summary.json").unlink(missing_ok=True)
    if not result.correlation_matrix.empty:
        result.correlation_matrix.to_csv(output / "correlation_matrix.csv", index_label="resource_request")
    else:
        (output / "correlation_matrix.csv").unlink(missing_ok=True)
    result.gpu_comparison.to_csv(output / "gpu_workload_comparison.csv", index=False)
    result.workload_size_summary.to_csv(output / "workload_size_summary.csv", index=False)
    result.missing_summary.to_csv(output / "missing_summary.csv", index=False)
    result.data_quality_summary.to_csv(output / "data_quality_summary.csv", index=False)

    figures = create_analytics_figures(result.classified_workloads)
    charts_dir = output / "charts"
    charts_dir.mkdir(exist_ok=True)
    for name, figure in figures.items():
        figure.write_html(charts_dir / f"{name}.html", include_plotlyjs="cdn")
    for existing in charts_dir.glob("*.html"):
        if existing.stem not in figures:
            existing.unlink()
    result.overview["generated_charts"] = [
        f"charts/{name}.html" for name in sorted(figures)
    ]
    (output / "overview.json").write_text(
        json.dumps(_json_clean(result.overview), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output / "analytics_report.md").write_text(_markdown_report(result), encoding="utf-8")
    LOGGER.info("Analytics reports written to %s", output)
    return result


def run_analytics(
    input_path: str | Path = PROCESSED_DATA_DIR / "cleaned_workloads.csv",
    output_dir: str | Path = REPORT_OUTPUT_DIR / "analytics",
) -> AnalyticsResult:
    """Load the cleaned data and generate its analytics report bundle."""
    source = Path(input_path)
    frame = load_cleaned_dataset(source)
    return generate_analytics(frame, output_dir, dataset_filename=source.name)
