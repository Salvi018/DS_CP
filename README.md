# GPUlytics

GPUlytics is a college-level project for analyzing GPU workloads, simulating cluster scheduling, predicting historical scheduling delays, and comparing resource allocation algorithms using the Alibaba GPU Cluster Dataset 2025.

## Planned features

- Dataset inspection and preprocessing
- Workload analytics
- Configurable simulated CPU/GPU clusters
- First Fit, Best Fit, and fragmentation-aware scheduling
- Simulation and evaluation
- Historical scheduling-delay prediction
- Interactive Streamlit dashboard
- Automated tests and deployment documentation

## Technology stack

Python, Pandas, NumPy, scikit-learn, Streamlit, Plotly, SQLite, Joblib, and Pytest. FastAPI may be introduced later if a separate API layer is needed.

## Installation

Python 3.11 or a compatible newer version is recommended. On macOS, from this directory:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, create and activate the environment with:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Launch the dashboard

From the project directory with the virtual environment activated:

```bash
streamlit run app.py
```

The dashboard uses a dark navy/teal theme configured in
`.streamlit/config.toml`, with consistent table, metric, chart, and action styling.

### Dashboard pages

- **Overview** reports raw/prepared data, module and model-artifact availability,
  the actual cleaned-record count, available workload/resource summaries, and a
  creation-time distribution when finite timestamps are present. Its chart labels
  source timestamp units as unverified; missing or invalid timestamps are not
  filled in.
- **Dataset Inspection** displays chunked inspection of the raw CSV, detected schema,
  data preview sample rows, numerical column statistics, categorical value counts,
  detected missing values, and data-quality findings.
- **Workload Analytics** offers reactive category and numeric range filters with
  dynamic sliders for request, limit, and timing columns found in the cleaned CSV.
  Filters are applied to the rows before summaries and charts are calculated; empty
  filter selections are handled gracefully.
- **Data Quality & Reports** displays existing reports without generating new
  data. Missing or malformed artifacts are reported as unavailable rather than
  being treated as successful results. Cleaning-report freshness is checked
  against the recorded raw-file path, size, and modification time.
- **Data Preparation** audits and triggers the clean preparation pipeline on
  explicit user action, displaying raw/cleaned rows, validation findings, and
  offering cleaning report downloads.
- **Cluster Simulator** provides discrete-event cluster simulation with an explicit
  run-button action. Simulates First Fit, Best Fit, or IFAS, tracking event
  history, queue delays, and placement decision diagnostics.
- **Scheduler Comparison** is the benchmark/evaluation view. It presents the saved
  evaluation results, same-input/fresh-cluster controls, and capacity invariant status;
  it only runs an experiment after the explicit button action.
- **AI Predictions** reads the saved model/evaluation artifacts, provides an
  interactive workload delay prediction form with input validation, and only trains
  after the explicit train/refresh action.
- **Algorithm Comparison** provides a dedicated, controlled side-by-side comparison
  of First Fit, Best Fit, and IFAS on identical ordered workloads with fresh cluster
  states. Displays invariant verification (0 capacity violations), outcome distributions,
  queue delay metrics, runtime comparisons, and downloadable comparison reports.

Dashboard reruns do not start a simulation, benchmark, or model-training job.
Corrupt or incompatible saved delay-model bundles produce a clear prediction
error instead of an unhandled deserialization/schema exception.

## Inspect and analyze the dataset

Place the dataset at `data/raw/disaggregated_DLRM_trace.csv`, then run from the
project directory:

```bash
python scripts/inspect_dataset.py
python scripts/inspect_dataset.py --input data/raw/disaggregated_DLRM_trace.csv
python scripts/inspect_dataset.py --skip-plots
```

The inspector uses the expected filename when present. If it is absent, a sole
CSV in `data/raw/` is selected and identified in the report; multiple candidates
require `--input`. Reports and applicable charts are written to `reports/eda/`.
The CSV is opened read-only and analyzed in chunks. Timestamp units remain
unverified unless confirmed from authoritative dataset documentation.

## Prepare cleaned workload data

Run the audited preprocessing pipeline from the project directory:

```bash
python scripts/prepare_dataset.py
python scripts/prepare_dataset.py --strict
python scripts/prepare_dataset.py --input data/raw/disaggregated_DLRM_trace.csv \
  --output data/processed/cleaned_workloads.csv \
  --report-dir reports/cleaning
```

The default output is `data/processed/cleaned_workloads.csv`; JSON and Markdown
audit summaries are written under `reports/cleaning/`. Row-level validation
issues and rejected rows are saved when applicable. Warnings (such as missing
timestamps and negative resource values) are non-fatal. Strict mode fails for
critical schema, numeric, and essential-identifier issues before replacing an
existing processed CSV. Raw CSVs are never modified.

Timestamp units are unverified. Timestamp source values and derived
`scheduling_delay_raw_units` / `observed_lifetime_raw_units` are retained in the
original units; no `*_sec` features are produced. Scheduling delay is a
historical target candidate and is excluded from the creation-time feature
list. Full-trace application-frequency and relative-arrival features are
descriptive only and excluded from prediction inputs.

## Workload analytics

Generate analytics from the cleaned workload dataset:

```bash
python scripts/analyze_workloads.py
python scripts/analyze_workloads.py --input data/processed/cleaned_workloads.csv \
  --output-dir reports/analytics
```

Outputs include overview, resource, role, application, timing, correlation,
GPU-requesting versus zero-GPU comparison, data-quality, workload-size, and
interactive Plotly chart reports under `reports/analytics/`. Workload-size
thresholds are transparent configurable analytical heuristics, not official
Alibaba labels. Scheduling and lifetime values remain in unverified source
timestamp units. The dashboard's Workload Analytics page reads cached results;
use its refresh button after changing the prepared dataset.

## Cluster simulation and baseline schedulers

Run a deterministic, discrete-event simulation on the configurable hypothetical
cluster in `config/demo_cluster.json`:

```bash
python scripts/run_simulation.py --max-workloads 1000
python scripts/run_simulation.py --input data/processed/cleaned_workloads.csv \
  --cluster-config config/demo_cluster.json --output-dir reports/simulations \
  --max-workloads 1000 --time-unit trace_units_unverified --seed 42 \
  --policy first_fit
python scripts/run_simulation.py --synthetic-demo --policy best_fit \
  --output-dir reports/simulation-demo
python scripts/run_simulation.py --max-workloads 1000 --compare-policies
python scripts/run_simulation.py --synthetic-demo --policy ifas \
  --ifas-config config/ifas.json --output-dir reports/simulation-ifas
```

The simulator validates each workload's complete request vector before
reserving resources, releases allocations on completion, retries queued
workloads, and writes its event/resource history and assumptions beneath
`reports/simulations/`. First Fit checks nodes in ascending node-ID order and
chooses the first accounting-feasible node. Best Fit chooses the feasible node
with the smallest sum of post-placement residual/capacity ratios across
positive-capacity resource dimensions (CPU, integral GPU devices, memory,
RDMA, disk, and configured extras); ties use ascending node ID. Dimensions
have equal weights. The score does not model topology, GPU architecture,
device fragmentation, future arrivals, or global optimality. GPU device count
is an independent integral dimension and is not a proxy for GPU memory or
compute capability.

Both policies use atomic accounting for every resource and compatibility
constraint. Pending workloads retry once after each release, in original queue
order; unchanged failures are not redundantly logged, while retry attempts are
counted. The comparison command runs identical ordered inputs against deep
copies of the same initial cluster configuration, and writes per-policy runs
plus comparison metrics beneath `reports/simulations/comparison/`. Queue delay
is arrival to first allocation for allocated workloads only. Scheduler runtime
measures time in policy selection. Unique pending failures and failed attempts
are separate metrics.

### IFAS score and fragmentation metrics

IFAS is a deterministic heuristic registered as `ifas`, configured in
`config/ifas.json` or via `--ifas-config`. Defaults:

```text
balance_weight       = 0.25
fragmentation_weight = 0.35
spread_weight        = 0.20
residual_fit_weight  = 0.20
enable_future_fit    = false
future_fit_weight    = 0.20 (used only when enabled)
tie_breaking         = node_id
```

For every accounting-feasible node, IFAS calculates normalized residual
capacity per positive-capacity dimension after hypothetical placement.
Zero-capacity dimensions with zero demand are omitted. The score is the
weighted average of:

- `balance_penalty`: population standard deviation of normalized residual
  fractions; lower values favor less imbalanced residual resources.
- `fragmentation_penalty`: when known queued workloads exist, zero if any
  role-compatible queued demand fits the candidate's post-placement residual
  vector; otherwise the mean normalized residual fraction. With no known
  queue, this term is zero and diagnostics note that it could not be estimated.
- `spread_penalty`: one when the candidate is idle while an already-active
  feasible node exists; otherwise zero.
- `residual_fit_penalty`: mean normalized residual fraction; lower values favor
  tighter fits.
- Optional `future_fit_penalty`: fraction of currently queued workloads that
  are feasible before the placement but would no longer fit any compatible
  node after it. Only the decision-time queue is considered; future trace
  arrivals are never inspected. It is disabled by default because of its
  additional queue-by-node checks.

The four base weights must be finite, non-negative, and not all zero; if
future-fit is enabled its weight must be positive. The score is divided by the
sum of enabled weights. Lowest score wins; ties use node ID or configured node
order. These score terms are normalized heuristics, not a global optimum.
Structured per-decision scores and rejected candidate reasons are written to
`scheduler_decisions.csv` (with detailed candidate scores controlled by
`diagnostics_enabled`).

The simulator also records event-time fragmentation snapshots and
time-weighted summaries. The representative GPU metric reports residual GPU
slots and the fraction stranded relative to the earliest queued GPU workload.
The multi-resource metric is the fraction of role-compatible queued workloads
for which summed residual resources are sufficient across nodes but no single
node can fit the entire vector. Aggregate capacity is only diagnostic: it
does not prove a workload is placeable. These metrics do not establish actual
GPU utilization.

`--max-workloads` selects the earliest rows with valid
arrival times, which is useful for controlled smoke runs; missing and invalid
timestamps are otherwise reported as exclusions. Missing observed lifetimes
are treated as censored (no completion event is invented). Timestamp values
stay in relative trace units; `--time-unit` is a report label and does not
convert values to seconds.

The example cluster is hypothetical and is not Alibaba hardware inventory.
Its resource capacities are expressed in the same request units as the trace.
The configuration explicitly maps `CN` and `HN` workload roles to matching
configured node roles, but this is an assumption, not a verified placement
rule. Application-density caps are disabled unless the cluster configuration
sets `density_limit_semantics_verified` true. Negative `max_instance_per_node`
values are rejected by default because their meaning is unverified; only an
explicit `density_unlimited_values` entry interprets a chosen integer as an
unlimited sentinel. Programmatic callers that omit a policy retain the legacy
`unique_feasible_node` behavior, which queues ambiguous placements.
Time-weighted allocation ratios describe reserved requests relative to
configured capacity, not measured GPU/CPU utilization, temperature, or power.

The Streamlit **Cluster Simulator** section runs only after the user selects
the run button and uses this fixed cleaned-data path and cluster configuration.

## Scheduler evaluation and benchmarks

Run a reproducible comparison using the same cleaned-workload subset, ordered
arrivals, resource constraints, compatibility configuration, and fresh cluster
state for each selected policy:

```bash
python scripts/benchmark_schedulers.py --max-workloads 1000
python scripts/benchmark_schedulers.py --policies first_fit best_fit ifas \
  --sample-size 1000 --seed 42
python scripts/benchmark_schedulers.py --synthetic-demo
python scripts/benchmark_schedulers.py --max-workloads 500 \
  --sensitivity --output-dir reports/benchmarks
```

Use `python scripts/benchmark_schedulers.py --help` for all arguments. Sampling
is deterministic for a given seed. Trace-replay selection requires valid,
non-negative observed arrivals; invalid timestamps and rows omitted by a
workload limit or sample are listed in `excluded_workloads.csv`. Invalid
resources, roles, and unverified density values from selected rows are also
retained as row-level exclusions. The CLI does not fill in absent arrivals or
completion times. Without a valid observed duration, an allocated workload has
no completion event and remains active when the finite event queue drains.

Each experiment writes to a timestamped child folder under
`reports/benchmarks/`. It includes the selected workload IDs, experiment and
cluster configuration, policy comparison, per-workload outcomes, queue-delay
and resource metrics, fragmentation snapshots, unique pending failure reasons,
row-level exclusions, an experiment log, Markdown summary, per-policy
simulation artifacts, and interactive Plotly HTML charts. The optional
`--sensitivity` mode runs one-factor-at-a-time scenarios for GPU/CPU/memory
capacity, GPU node count, positive density-cap enforcement (clearly marked as
an assumption when unverified), IFAS weights when IFAS is selected, and source
ordering within equal-arrival-time groups when ties exist. Unequal observed
arrival times are not reordered. Every scenario configuration and result is
saved separately; it does not silently change the baseline cluster.

The event loop stops when its finite arrival/completion event heap is empty.
Pending workloads retry once after each completion/release event; an
unplaceable workload remains pending and does not cause an infinite retry loop.
Deterministic policies are run once even if `--repeats` requests multiple
trials; the requested count and the reason no redundant repetitions were run
are recorded. Scheduler wall-clock time is measured separately from simulated
trace time and is not a correctness assertion.

Queue delay is defined only as simulated arrival to first successful
allocation; it excludes pending and rejected workloads and is not historical
creation-to-scheduling delay. Resource ratios describe requested-resource
reservations and not hardware utilization. Zero-capacity dimensions are
reported as not applicable. Fragmentation values are event-snapshot diagnostics
over currently queued work: GPU-fit metrics separate aggregate-capacity
shortfalls from cases with enough aggregate resources but no single-node fit.
Role-incompatible jobs are excluded from the fragmentation denominators.
Aggregate resource sufficiency does not prove a valid placement, and these
metrics do not incorporate unverified density semantics.

The dashboard **Scheduler Comparison** page has explicit workload, policy, and
sensitivity controls. It does not launch experiments on reruns.

## Historical scheduling-delay prediction

Train and evaluate the delay pipeline from the cleaned dataset:

```bash
python scripts/train_delay_model.py
python scripts/train_delay_model.py --input data/processed/cleaned_workloads.csv \
  --output-dir reports/ml --model-dir models --seed 42 --test-size 0.2 \
  --model auto
```

The target is recomputed as `scheduled_time - creation_time`; negative delays
and rows with missing/invalid timestamps are excluded with row-level reasons.
Timestamp units remain **unverified source units**, not seconds. The
chronological split groups by distinct creation times and partitions train,
validation, and test in temporal order. Validation MAE chooses the model;
the held-out test set is used only for final evaluation. Numeric imputation,
scaling, and categorical encoding are fitted inside the training pipeline.

The explicit creation-time allowlist consists of CPU, GPU, memory, RDMA, and
disk requests, `creation_time`, `role`, and `app_name`, when present. The
pipeline excludes scheduled/deletion timestamps, delay/lifetime/relative-arrival
outcomes, whole-trace application frequency, and `max_instance_per_node` while
density semantics remain unverified. Unknown application categories are
handled by the fitted one-hot encoder. The selected complete preprocessing and
estimator pipeline is saved at
`models/scheduling_delay_pipeline.joblib`; its feature schema, training
configuration, held-out metrics, prediction examples, exclusions, diagnostics,
and report are saved under `reports/ml/`.

The dashboard **AI Predictions** page exposes an explicit train/refresh action
and an inference form. For programmatic inference after training:

```python
from src.ml_prediction import predict_scheduling_delay

prediction = predict_scheduling_delay(
    {
        "creation_time": 1000,
        "role": "HN",
        "app_name": "example-app",
        "cpu_request": 4,
        "gpu_request": 1,
        "memory_request": 16,
        "rdma_request": 1,
        "disk_request": 20,
    },
    "models/scheduling_delay_pipeline.joblib",
)
print(prediction)
```

The output is an estimate of historical delay, not a guaranteed outcome, actual
GPU utilization, or the effect of any scheduling policy. The training report
must be checked before interpreting results; candidate models may perform worse
than the median baseline.

## Dataset

Download the Alibaba GPU Cluster Dataset 2025 from [Kaggle](https://www.kaggle.com/datasets/mdsultanulislamovi/alibaba-gpu-cluster-dataset-2025). The dataset is not downloaded automatically and is excluded from Git.

Place the expected CSV at:

```text
data/raw/disaggregated_DLRM_trace.csv
```

The dashboard starts without the dataset and explains where to place it. The existing workspace may contain a copy outside the project folder; copy it into the path above if you want the dashboard to detect it.

## Current implementation status

The project includes configuration, a Streamlit dashboard with cached dataset
inspection and analytics, user-triggered preparation and simulation, a chunked
dataset-inspection/EDA command, an audited cleaning pipeline, workload
analytics, deterministic First Fit and Best Fit policies, and the heuristic
IFAS scheduler with atomic accounting, controlled comparisons, and a
reproducible scheduler-evaluation framework. The historical scheduling-delay
prediction pipeline is implemented with a chronological evaluation split,
serialized preprocessing, CLI, inference API, report artifacts, and an explicit
dashboard training action. Persistence beyond model artifacts and API
functionality are not implemented. No sample statistics or results are
fabricated.
