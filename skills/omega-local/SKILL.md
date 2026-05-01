---
name: omega-local
description: >
  Generate time-series embeddings locally using a vendored Omega encoder
  checkpoint — no Newton API, no cloud roundtrip. Use when the user wants
  to run the Omega 1.3 encoder offline (PyTorch + a `.pt` checkpoint),
  build a custom classification or anomaly-detection pipeline on top of
  the embeddings (KNN, Isolation Forest, t-SNE/UMAP), or experiment with
  windowing/normalization strategies that the cloud Lens API doesn't expose.
  This skill covers checkpoint loading, single-channel input shape,
  windowing + padding, normalization choices, the "joint state" pattern
  for multi-sensor data, and common pitfalls.
  Do NOT use for the cloud Newton API (use newton-setup, newton-machine-state,
  etc.). Do NOT use for vision tasks (no image encoder is exposed locally).
---

# Omega Local Inference

Run the Omega 1.3 encoder directly in PyTorch from a checkpoint file. This is the offline counterpart to the cloud Machine State Lens — same encoder, but you control windowing, normalization, and the downstream classifier instead of using Newton's hosted KNN.

## When to Apply

- User has an Omega checkpoint (`.pt`) and wants to generate embeddings without hitting the API
- User is prototyping a custom downstream model (KNN with custom metric, Isolation Forest, neural classifier, etc.) on top of Omega embeddings
- User wants to inspect raw embeddings (PCA / t-SNE / UMAP) before deciding on a classifier
- User needs reproducible offline experiments (paper, benchmark, air-gapped environment)
- User is migrating an existing notebook from another encoder to Omega

**Use Newton Machine State Lens instead when**: the user just wants classification results, doesn't need to control the model, and is happy with a hosted KNN. The local path adds significant code complexity in exchange for control.

## What You Need

| Thing | Notes |
|-------|-------|
| Omega checkpoint | `.pt` file, e.g. `omega1_3_encoder_forecast_decoder.pt`. Place under `omega_core/weights/` (or wherever convenient). |
| `omega_core/` SDK | Vendored package containing `core/`, `embedding_lens.py`, `model_loader.py`, `utils.py`. Treat `omega_core/core/` as upstream code — do not edit it locally; sync changes through the source of truth instead. |
| Python 3.11 or 3.12 | 3.9 is EOL and several deps (numpy 2.x, pandas 3.x) require 3.10+. |
| Deps | `torch>=2.0`, `numpy`, `pandas`, `scikit-learn`, `einops`, `umap-learn`, `plotly` |

## Encoder Quick Facts

- **Embedding dim**: 768 (CLS token at position `[:, -1, :]` of encoder output)
- **Target series length**: 1024 timesteps
- **Channels**: **1** — Omega is single-channel. Multi-sensor data is processed per-sensor; embeddings are concatenated downstream (the "joint state" pattern below).
- **Encoder input format**: a dict `{"timeseries": tensor(B, 1, T), "timepoint_mask": tensor(B, T)}`

## Loading the Encoder

```python
from omega_core.embedding_lens import EmbeddingLens

lens = EmbeddingLens(
    checkpoint_path="omega_core/weights/omega1_3_encoder_forecast_decoder.pt",
    device=None,            # auto-detects cuda/mps/cpu
    normalize_input=True,   # see "Normalization choices" below
)

embedding = lens.embed(timeseries)   # shape (1, 1024) → (768,)
```

`get_encoder_from_checkpoint(path)` (in `omega_core/model_loader.py`) is the lower-level loader if you need direct access to the `nn.Module`.

## Single-Channel Constraint

Omega's `in_channels=1`. **Do not** stack sensors along the channel axis. The supported pattern is:

1. Treat each sensor column as an independent univariate series.
2. Run the encoder per-sensor per-window → one 768-dim embedding per (sensor, window).
3. Concatenate embeddings across sensors at the **window** level for the downstream model (joint state).

Trying to feed `(B, n_sensors, T)` raises `ValueError: Model only supports single-channel input`.

## Windowing

For `T = window_size` rows of a single sensor:

- `window_size == 1024` → fast path, encoder runs as-is.
- `window_size < 1024` → **left-pad** with zeros to 1024 and pass a `timepoint_mask` (1.0 for valid points, 0.0 for padding). Padding goes on the **left** (start of the sequence), valid points are the most recent. Instance normalization (when enabled) is computed only over valid points.
- `window_size > 1024` → not supported. Resample, downsample, or split.

The reference loop (with overlap):

```python
read_indexes = list(range(0, num_rows - window_size + 1, step_size))
for read_idx in read_indexes:
    window = sensor_data[read_idx : read_idx + window_size]
    emb = lens.embed(window.reshape(1, -1))   # → (768,)
```

A common starting point: `window_size=1024, step_size=512` (50% overlap). For short datasets (a few hundred rows), use smaller windows (e.g. 30 minutes for 1-min sampling) with heavy overlap (`step=5`).

## Normalization Choices

The encoder was trained with instance-normalized inputs (per-window mean 0, std 1). You have two options:

### Option A — let the encoder normalize (`normalize_input=True`)

Each window is independently centered and scaled inside `EmbeddingLens.embed()`. Simple, matches training distribution. **Downside**: every window looks "self-similar" — a window with values 0–1 and a window with values 100–200 produce nearly identical embeddings. You lose cross-window context.

### Option B — normalize externally, set `normalize_input=False`

Z-score each sensor across the **whole dataset** with `StandardScaler`, then disable instance norm. The encoder sees globally normalized values, so a low-amplitude window stays "low" relative to the dataset.

```python
from sklearn.preprocessing import StandardScaler

df_norm = df.copy()
df_norm[sensor_cols] = StandardScaler().fit_transform(df[sensor_cols])

lens = EmbeddingLens(checkpoint_path=..., normalize_input=False)
```

This is usually the right choice for anomaly detection and any task where the absolute level of a signal matters. Fit the scaler **on train only** and reuse it for test/inference — fitting on the whole dataset leaks information.

## The "Joint State" Pattern

Per-window output from the encoder is one 768-dim vector per sensor. To get a single feature vector representing the whole machine at that window, **concatenate across sensors in a stable order**:

```python
import pandas as pd
import numpy as np

# df_emb: long format, one row per (sensor, window)
# columns: sensor, read_index, embedding (np.ndarray of length 768), ...
sensor_order = sorted(df_emb["sensor"].unique())   # alphabetical or explicit list
pivot = df_emb.pivot(index="read_index", columns="sensor", values="embedding")[sensor_order]
X = np.stack([np.concatenate(row) for row in pivot.values])
# X.shape == (n_windows, n_sensors * 768)
```

Two requirements:
- Same `sensor_order` at train, test, and inference time (or your KNN distances will be meaningless).
- Every window must contain every sensor — drop windows where a sensor is missing.

## Preflight: data-quality checks before embedding

The Omega encoder works best on **monotonically increasing timestamps with a roughly constant sampling rate**. Deviations don't always break the pipeline, but they're worth flagging early — and the same checks catch unrelated data issues that *do* break things downstream.

Pattern adapted from [`archetypeai/omega-1-4-preflight`](https://github.com/archetypeai/omega-1-4-preflight) (that repo targets the cloud `machine-state-job-pipeline` for `omega_1_4_base`; the static checks transfer cleanly to local 1.3 inference). Run once after loading, before any transformations:

| Check | What it catches | Severity |
|---|---|---|
| `schema` | Time column missing, non-numeric features | FAIL |
| `timestamp` | Non-monotonic, duplicates, gaps where `max(Δt) > N × median(Δt)` | FAIL/WARN |
| `missing_values` | NaN columns (worst-offender first) | WARN |
| `constant_columns` | Zero-variance sensors that should be dropped | WARN |
| `feature_scale` | Range gap > N decades — flags need for StandardScaler | WARN |
| `class_balance` | Severe (>85%) or moderate (>70%) imbalance, with majority baseline | WARN |
| `window_vs_sampling` | Translates `window_size × median(Δt)` into human time so you can sanity-check it covers a typical fault duration | INFO |

Sketch:

```python
def preflight(df, time_col, sensor_cols=None, label_col=None, block_col=None,
              window_size=None, gap_threshold_x=5.0, scale_decade_warn=3.0):
    """Returns list of (level, name, message). Use block_col for per-block
    timestamp checks (e.g. when each job is a separate recording)."""
    results = []

    # 1. schema (time col, numeric sensors)
    # 2. timestamp per block: monotonic, unique, max(Δt)/median(Δt) ≤ gap_threshold_x
    # 3. missing values per column
    # 4. constant columns: std < 1e-12
    # 5. feature-scale heterogeneity: log10(range) span > scale_decade_warn
    # 6. class balance with majority baseline
    # 7. window_size × median(Δt) → human-readable duration

    return results
```

Run with `block_col='job_id'` (or whatever splits your recordings) when the dataset has multiple independent blocks — windowing must not cross block boundaries, and per-block timestamp checks catch issues a global check would hide.

## Pipeline Skeleton

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from omega_core.embedding_lens import EmbeddingLens

# 1. Load + clean
df = pd.read_csv("data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# 2. Drop constant sensors (std=0 → no information; can also poison normalization)
sensors = [c for c in df.columns if c not in ("timestamp", "label")]
sensors = [c for c in sensors if df[c].std() > 0]

# 3. Normalize (fit on train only in production)
scaler = StandardScaler().fit(df[sensors])
df[sensors] = scaler.transform(df[sensors])

# 4. Embeddings, per sensor per window
lens = EmbeddingLens(checkpoint_path="weights/omega.pt", normalize_input=False)
W, S = 60, 30   # window_size, step_size — pick to fit your dataset

records = []
for col in sensors:
    series = df[col].values.astype(np.float32)
    for read_idx in range(0, len(series) - W + 1, S):
        window = series[read_idx:read_idx + W]                    # shape (W,)
        emb = lens.embed(window.reshape(1, -1))                   # → (768,)
        label = df["label"].iloc[read_idx + W - 1]                # label of last point
        records.append({"sensor": col, "read_index": read_idx,
                        "embedding": emb, "label": label})

df_emb = pd.DataFrame(records)

# 5. Joint state
pivot = df_emb.pivot(index="read_index", columns="sensor", values="embedding")[sensors]
X = np.stack([np.concatenate(row) for row in pivot.values])
y = df_emb.groupby("read_index")["label"].first().reindex(pivot.index).values

# 6. Out-of-time split (no shuffling — preserve temporal order)
n_train = int(0.7 * len(X))
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# 7. Downstream model
clf = KNeighborsClassifier(n_neighbors=3, metric="manhattan", weights="distance")
clf.fit(X_train, y_train)
print("test acc:", clf.score(X_test, y_test))
```

For anomaly detection, swap step 7 for `IsolationForest(contamination="auto").fit(X_train)` and don't pass `y_train` — labels are only used for evaluation. **Caveat**: Isolation Forest assumes anomalies are *rare and structurally different*. If your "anomaly" class is 30%+ of data and shares structure with normal (e.g. same machine, different operating mode), IF will collapse to predicting "all normal." That's a classification problem, not an anomaly-detection one — stick with the supervised baseline.

## Beyond a baseline: threshold tuning + model selection

The default KNN baseline often underperforms on the rare/expensive class — even when overall accuracy looks fine. Two leverage points before reaching for a different architecture:

### 1. Decision-threshold tuning

sklearn defaults to threshold 0.5 on `prob_1`. For binary fault detection, a lower threshold catches more faults at the cost of more false alarms. Sweep and pick:

```python
prob_fault = clf.predict_proba(X_test)[:, 1]
for t in np.linspace(0.05, 0.95, 19):
    y_pred = (prob_fault >= t).astype(int)
    # compute precision, recall, F-beta — pick the t that fits your cost ratio
```

### 2. F-beta scoring

When recall on the rare class matters more than precision, score with F-beta instead of F1:

```python
def fbeta(precision, recall, beta=2.0):
    if precision + recall == 0: return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)
```

`beta=1` is F1 (balanced). `beta=2` weights recall 2× as much as precision — appropriate when missing a fault is much more costly than a false alarm. `beta=4` heavily favors recall; `beta=0.5` favors precision.

### 3. Multi-model auto-select

Sweep across model variants, pick the F-beta-optimal threshold per model, then pick the F-beta-best overall:

```python
candidates = {
    'KNN distance':    KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance'),
    'KNN uniform':     KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='uniform'),
    'LogReg balanced': LogisticRegression(class_weight='balanced', max_iter=2000),
    'RF balanced':     RandomForestClassifier(class_weight='balanced', n_estimators=200,
                                              random_state=42, n_jobs=-1),
}
# For each: fit, sweep thresholds, record F-beta-best. Pick winner across models.
```

`class_weight='balanced'` (LogReg, RF) automatically reweights losses inversely to class frequency. KNN doesn't support `class_weight` natively — `weights='uniform'` vs `'distance'` is the closest knob, and threshold sweeping does more for KNN than weight choice.

Cross-cutting observation: on hard datasets, **RF balanced + low threshold** consistently dominates F2. KNN distance is a strong default for clean datasets where threshold ≈ 0.5 already works.

## Common Pitfalls

### 1. Constant-zero sensor columns

Sensors like `refVelocityBack` that are 0.0 across the whole dataset have `std=0`. Two failure modes:

- With `normalize_input=True`, the encoder's instance norm divides by `std + 1e-5`, gives zero output, and produces a degenerate constant embedding. Not strictly an error, but adds zero signal and inflates feature dimensionality.
- `StandardScaler` is robust (it returns zeros for constant features), but **manual `(x - mean) / std`** is not — that gives `0 / 0 = NaN`, which propagates into the encoder and out into the embeddings, then into t-SNE/KNN/IsolationForest, all of which raise `ValueError: Input X contains NaN`.

**Fix**: drop them up front: `sensors = [c for c in sensors if df[c].std() > 0]`.

### 2. NaN propagation is silent until the classifier

The encoder happily produces NaN embeddings for NaN input. Downstream visualizers and sklearn estimators reject NaN with a long traceback **far from the source**. Sanity-check after the embedding step:

```python
n_nan = sum(1 for e in df_emb["embedding"] if np.isnan(e).any())
assert n_nan == 0, f"{n_nan} NaN embeddings — check for NaN columns in df"
```

If you see exactly `n_windows` NaN embeddings, exactly one sensor column is bad — find it with `df.isna().sum()`.

### 3. Mixed-up sensor sets across runs

The joint state vector's dimensionality is `n_sensors * 768`. If you train a classifier on `X.shape == (N, 14*768)` and at inference time the data has 13 sensors (or 15), the classifier will fail with a shape mismatch — or worse, silently give garbage results if dimensions match by coincidence.

Pin `sensor_order` explicitly and assert `len(sensor_order)` at every stage. A common mistake is filtering sensors based on `df.columns` after a DataFrame mutation that added a column (e.g. `block_id`, `imputed`, or a leftover `label_num` from a copy-pasted notebook cell).

### 4. Notebook state corruption

Embeddings are expensive enough that people resist re-running cells. The result is `df_emb` from a previous configuration sitting in the kernel while downstream cells use new parameters. **When debugging unexpected NaN or shape errors, restart the kernel and Run All before assuming the code is wrong.**

### 5. window_size > 1024

Not supported — the encoder's positional embeddings are fixed at 1024. Resample (`df.resample("5min").mean()`) or split into multiple windows and aggregate embeddings.

### 6. Channel-axis confusion

If you call `lens.embed(np.random.randn(2, 1024))` (2 channels), you get `ValueError: Model only supports single-channel input`. Reshape to `(1, 1024)` per call, or pass a list of `(1, 1024)` arrays for batched inference.

### 7. Padding direction

Short windows are **left-padded**, not right-padded. The valid points are the *end* of the sequence. If you implement custom padding, match this convention or your embeddings won't be comparable to the rest of the pipeline.

### 8. OOT split breaks on single-transition datasets

Out-of-time (chronological) splitting assumes both classes appear throughout time. When the fault region is one contiguous block (e.g. rows 0–93 are fault, 94–246 are normal), a 70/30 split puts all of one class in train and all of the other in test:

- `y_test` has only one unique value — precision/recall on the missing class are undefined.
- Macro F1 may report 1.0 if the model trivially predicts the only present class — looks great but is uninformative.

The deceptive symptom: the pipeline runs without error and reports a strong-looking score. Always check `np.unique(y_train, return_counts=True)` and `np.unique(y_test, return_counts=True)` after splitting.

**Fix**: use `mode='random'` (with `random_state` for reproducibility) when the dataset has a single fault episode. OOT remains correct for multi-block datasets where each block is internally mixed (e.g. one job containing both fault and normal samples) — the issue is *temporal class structure*, not OOT vs random in general.

## Suggested Module Layout

A clean way to organize a project around these patterns is one class per file, each owning a single stage of the pipeline:

| Class | Responsibility |
|-------|----------------|
| `DataPreprocessor` | Splits a multivariate series with NaN gaps into continuous "blocks", imputes short gaps |
| `EmbeddingGenerator` | Wraps `EmbeddingLens`, handles per-sensor windowing, padding+mask for short windows, label propagation |
| `FeaturePreparer` | Pivots to joint-state matrix, normalization (L2/standardize), optional PCA |
| `DataSplitter` | Out-of-time or random train/test split |
| `MachineState` | Unified KNN (classify) / Isolation Forest (anomaly) with consistent prediction DataFrame |
| `Visualizer` | PCA/t-SNE/UMAP scatter, raw signals, predictions-over-time with label-shaded background |

Each class consumes the output of the previous one. The whole pipeline fits in ~1000 lines and is easy to swap individual stages on (e.g. trying a different windowing strategy without touching the classifier).

## Comparing Local vs Newton Machine State Lens

| Concern | Local Omega | Newton Machine State Lens |
|---------|-------------|---------------------------|
| Where the encoder runs | Your machine (CPU/MPS/CUDA) | Newton cloud |
| Auth | None — checkpoint file | Bearer token, see `newton-setup` |
| Classifier | Your choice — KNN, IF, MLP, anything | Hosted KNN over n-shot focus files |
| Window/step config | Full control via your loop | `csv_configs.window_size` / `step_size` |
| Normalization control | Pre-encoder StandardScaler or instance norm | Instance norm only |
| Multi-sensor handling | Joint-state concat (your code) | `data_columns` + per-stage sessions, see `newton-machine-state` |
| Cost | Free (compute only) | Per-inference billing |
| Real-time streaming | You build it | SSE consumer built-in |
| Reproducibility | Pinned by checkpoint hash + seed | Tied to API version |

If you need both — e.g. local prototyping followed by cloud deployment — the same `data_columns`, `window_size`, and `step_size` settings transfer cleanly. The encoder is the same on both sides; only the classifier and orchestration differ.
