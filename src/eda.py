"""Generate reproducible, schema-aware reports and plots from inspected traces."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dataset_inspector import (
    RESOURCE_NAMES,
    DatasetInspection,
    DatasetInspectionError,
    inspect_dataset,
)

LOGGER = logging.getLogger(__name__)
RESOURCE_CHART = "resource_distributions.png"
ROLE_CHART = "role_distribution.png"
APPLICATION_CHART = "top_applications.png"
DELAY_CHART = "scheduling_delay_distribution.png"
LIFETIME_CHART = "lifetime_distribution.png"
CORRELATION_CHART = "resource_correlation.png"
REQUEST_LIMIT_CHART = "resource_request_vs_limit.png"


def _save_resource_distributions(inspection: DatasetInspection, output: Path) -> bool:
    columns = [
        f"{resource}_request"
        for resource in RESOURCE_NAMES
        if f"{resource}_request" in inspection.sample.columns
    ]
    if not columns:
        return False
    rows = (len(columns) + 1) // 2
    figure, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows), squeeze=False)
    for axis, column in zip(axes.flat, columns):
        values = pd.to_numeric(inspection.sample[column], errors="coerce").dropna()
        if values.empty:
            axis.set_title(f"{column} (no valid numeric values)")
            continue
        axis.hist(values, bins=min(40, max(10, int(np.sqrt(len(values))))), color="#2471A3")
        axis.set_title(f"{column} distribution")
        axis.set_xlabel(column.replace("_", " ").title())
        axis.set_ylabel("Instance count (sample)")
        axis.grid(axis="y", alpha=0.25)
    for axis in axes.flat[len(columns) :]:
        axis.remove()
    figure.suptitle("Resource request distributions")
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return True


def _save_category_plot(
    inspection: DatasetInspection, column: str, title: str, output: Path
) -> bool:
    if column not in inspection.categorical_summary["column"].values:
        return False
    counts = inspection.categorical_summary.query("column == @column").head(20)
    if counts.empty:
        return False
    figure, axis = plt.subplots(figsize=(11, max(4, 0.35 * len(counts))))
    axis.barh(counts["category"].astype(str)[::-1], counts["count"][::-1], color="#2471A3")
    axis.set_title(title)
    axis.set_xlabel("Instance count")
    axis.set_ylabel(column.replace("_", " ").title())
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return True


def _save_delta_plot(
    inspection: DatasetInspection, first: str, second: str, title: str, output: Path
) -> bool:
    if first not in inspection.sample.columns or second not in inspection.sample.columns:
        return False
    left = pd.to_numeric(inspection.sample[first], errors="coerce")
    right = pd.to_numeric(inspection.sample[second], errors="coerce")
    values = (right - left).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return False
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(values, bins=min(50, max(10, int(np.sqrt(len(values))))), color="#148F77")
    axis.set_title(title)
    axis.set_xlabel(f"{second} - {first} (timestamp units unverified)")
    axis.set_ylabel("Instance count (sample)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return True


def _save_request_limit_plot(inspection: DatasetInspection, output: Path) -> bool:
    resources = [
        resource
        for resource in RESOURCE_NAMES
        if f"{resource}_request" in inspection.sample.columns
        and f"{resource}_limit" in inspection.sample.columns
    ]
    if not resources:
        return False
    figure, axes = plt.subplots(1, len(resources), figsize=(5 * len(resources), 4), squeeze=False)
    for axis, resource in zip(axes.flat, resources):
        request = pd.to_numeric(inspection.sample[f"{resource}_request"], errors="coerce")
        limit = pd.to_numeric(inspection.sample[f"{resource}_limit"], errors="coerce")
        valid = request.notna() & limit.notna() & np.isfinite(request) & np.isfinite(limit)
        axis.scatter(request[valid], limit[valid], alpha=0.35, s=12)
        axis.set_title(resource.upper())
        axis.set_xlabel("Request")
        axis.set_ylabel("Limit")
        axis.grid(alpha=0.25)
    figure.suptitle("Resource requests versus limits (sample)")
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return True


def _save_correlation_plot(inspection: DatasetInspection, output: Path) -> bool:
    columns = [
        column
        for column in inspection.sample.columns
        if column in inspection.numerical_summary["column"].values
        and inspection.sample[column].notna().any()
    ]
    if len(columns) < 2:
        return False
    numeric = inspection.sample[columns].apply(pd.to_numeric, errors="coerce")
    correlations = numeric.corr(min_periods=2)
    if correlations.empty or not correlations.notna().any().any():
        return False
    figure, axis = plt.subplots(figsize=(max(7, len(columns) * 0.65), max(6, len(columns) * 0.55)))
    image = axis.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(len(columns)), labels=columns, rotation=70, ha="right")
    axis.set_yticks(range(len(columns)), labels=columns)
    axis.set_title("Numerical correlations (sample; association, not causation)")
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return True


def _correlation_matrix(inspection: DatasetInspection) -> pd.DataFrame:
    columns = [
        column
        for column in inspection.sample.columns
        if column in inspection.numerical_summary["column"].values
    ]
    if len(columns) < 2:
        return pd.DataFrame()
    return inspection.sample[columns].apply(pd.to_numeric, errors="coerce").corr(min_periods=2)


def _markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No applicable columns or values were observed._"
    displayed = frame.head(max_rows).fillna("")
    headers = [str(column) for column in displayed.columns]
    rows = [
        [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        for row in displayed.itertuples(index=False, name=None)
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    if len(frame) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join(lines)


def _write_markdown_report(
    inspection: DatasetInspection,
    output_dir: Path,
    generated_charts: list[str],
    skipped_charts: dict[str, str],
    skip_plots: bool,
) -> None:
    report = inspection.report
    columns = pd.DataFrame(report["columns"])
    quality = inspection.data_quality
    quality_text = (
        _markdown_table(quality, max_rows=100)
        if not quality.empty
        else "_No data-quality anomalies were detected by the implemented checks._"
    )
    correlation = _correlation_matrix(inspection)
    chart_text = (
        "\n".join(f"- [{name}]({name})" for name in generated_charts)
        if generated_charts
        else "_No charts generated._"
    )
    skipped_text = (
        "\n".join(f"- `{name}`: {reason}" for name, reason in skipped_charts.items())
        if skipped_charts
        else "_No charts were skipped._"
    )
    content = f"""# GPUlytics dataset inspection and EDA

- **Dataset:** `{report['dataset_filename']}` (`{report['dataset_path']}`)
- **Inspection time (UTC):** {report['inspection_timestamp_utc']}
- **Dimensions:** {report['row_count']:,} rows × {report['column_count']} columns
- **Timestamp unit assumption:** {report['timestamp_analysis']['unit_assumption']}

## Observed columns and data types

{_markdown_table(columns, max_rows=100)}

## Schema comparison

- Missing expected columns: {', '.join(f'`{name}`' for name in report['missing_expected_columns']) or 'None'}
- Extra observed columns: {', '.join(f'`{name}`' for name in report['extra_columns']) or 'None'}
- Possible renamed fields based on name similarity only: {json.dumps(report['possible_renames'])}
- A possible rename is a heuristic and requires human verification.

## Missingness

{_markdown_table(inspection.missing_values.sort_values('missing_percent', ascending=False))}

Missing timestamp values can reflect incomplete observation windows and are not automatically treated as corruption.

## Data-quality findings

{quality_text}

Duplicate full rows: **{report['duplicate_rows']:,}**  
Duplicate non-missing instance identifiers: **{report['duplicate_instance_identifiers']:,}**  
Zero GPU requests: **{report['zero_gpu_request_count']:,}** (not necessarily anomalous; CPU-only workloads may be valid).

## Numerical summaries

{_markdown_table(inspection.numerical_summary)}

{report['statistics_note']}

## Categorical summaries

The complete category frequency tables, including instance counts by GPU request value, are in `categorical_summary.csv`. This report shows up to 30 rows.

{_markdown_table(inspection.categorical_summary)}

## Numerical correlations

Pearson correlations are calculated over the bounded observed-row sample, where at least two paired numeric values exist. Correlation describes association, not causation.

{_markdown_table(correlation.reset_index(names="column"))}

## Timestamp analysis

```json
{json.dumps(report['timestamp_analysis'], indent=2)}
```

Timestamp values are parsed numerically. Units remain unverified unless confirmed from dataset documentation. Negative delays and lifetimes are reported and retained; no rows are silently discarded.

## Charts generated

{chart_text}

## Charts skipped

{skipped_text}

## Limitations and recommended next steps

- This inspection is descriptive and does not establish causal relationships between numerical variables.
- Numerical quantiles, timestamp distribution summaries, and chart data use bounded leading-row samples; streamed counts, means, standard deviations, minima, maxima, duplicate counts, and anomaly counts use all rows.
- Role and `max_instance_per_node` domains are not sufficiently specified in the available schema description; observed values are listed without inventing an allowed-value set.
- Confirm timestamp units and workload-domain value conventions against the dataset's authoritative documentation before downstream interpretation.
- Review schema mismatches and negative values with domain documentation before cleaning or modeling.
"""
    (output_dir / "eda_report.md").write_text(content, encoding="utf-8")


def run_eda(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    skip_plots: bool = False,
    chunksize: int = 50_000,
    sample_size: int = 10_000,
) -> DatasetInspection:
    """Inspect a CSV and write structured reports plus applicable EDA plots."""
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    inspection = inspect_dataset(input_path, chunksize=chunksize, sample_size=sample_size)

    inspection.schema.to_csv(destination / "schema_report.csv", index=False)
    inspection.missing_values.to_csv(destination / "missing_values.csv", index=False)
    inspection.numerical_summary.to_csv(destination / "numerical_summary.csv", index=False)
    inspection.categorical_summary.to_csv(destination / "categorical_summary.csv", index=False)
    inspection.data_quality.to_csv(destination / "data_quality_report.csv", index=False)
    correlations = _correlation_matrix(inspection)
    if not correlations.empty:
        correlations.to_csv(destination / "resource_correlation.csv", index_label="column")
    else:
        (destination / "resource_correlation.csv").unlink(missing_ok=True)

    generated_charts: list[str] = []
    skipped_charts: dict[str, str] = {}
    chart_jobs: list[tuple[str, Any, str]] = [
        (
            RESOURCE_CHART,
            lambda: _save_resource_distributions(inspection, destination / RESOURCE_CHART),
            "No resource request columns exist.",
        ),
        (
            ROLE_CHART,
            lambda: _save_category_plot(inspection, "role", "Workload count by role", destination / ROLE_CHART),
            "The role column or non-missing role values are unavailable.",
        ),
        (
            APPLICATION_CHART,
            lambda: _save_category_plot(
                inspection, "app_name", "Top 20 applications by instance count", destination / APPLICATION_CHART
            ),
            "The app_name column or non-missing application values are unavailable.",
        ),
        (
            REQUEST_LIMIT_CHART,
            lambda: _save_request_limit_plot(inspection, destination / REQUEST_LIMIT_CHART),
            "No resource request/limit pair exists.",
        ),
        (
            DELAY_CHART,
            lambda: _save_delta_plot(
                inspection, "creation_time", "scheduled_time", "Scheduling delay distribution",
                destination / DELAY_CHART
            ),
            "Valid creation_time and scheduled_time pairs are unavailable.",
        ),
        (
            LIFETIME_CHART,
            lambda: _save_delta_plot(
                inspection, "scheduled_time", "deletion_time", "Observed scheduled lifetime distribution",
                destination / LIFETIME_CHART
            ),
            "Valid scheduled_time and deletion_time pairs are unavailable.",
        ),
        (
            CORRELATION_CHART,
            lambda: _save_correlation_plot(inspection, destination / CORRELATION_CHART),
            "Fewer than two suitable numerical columns are available.",
        ),
    ]
    for filename, generate, reason in chart_jobs:
        if skip_plots:
            skipped_charts[filename] = "Plot generation disabled by --skip-plots."
            (destination / filename).unlink(missing_ok=True)
            continue
        try:
            if generate():
                generated_charts.append(filename)
            else:
                skipped_charts[filename] = reason
                (destination / filename).unlink(missing_ok=True)
        except (ValueError, TypeError) as exc:
            LOGGER.warning("Skipping chart %s: %s", filename, exc)
            skipped_charts[filename] = f"Plot could not be generated: {exc}"
            (destination / filename).unlink(missing_ok=True)

    inspection.report["generated_charts"] = generated_charts
    inspection.report["skipped_charts"] = skipped_charts
    inspection.report["correlations_sample"] = (
        json.loads(correlations.to_json()) if not correlations.empty else {}
    )
    (destination / "dataset_summary.json").write_text(
        json.dumps(inspection.report, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_markdown_report(
        inspection, destination, generated_charts, skipped_charts, skip_plots
    )
    return inspection
