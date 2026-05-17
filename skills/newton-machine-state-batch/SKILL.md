---
name: newton-machine-state-batch
description: >
  Run the Machine State classification pipeline as an asynchronous batch
  job on the Archetype AI Newton platform — classify millions of rows of
  time-series sensor data into n-shot example classes without holding a
  WebSocket session open. Powered by an Omega encoder (default
  `omega_1_4_base` — generic, arbitrary channel counts) plus a hosted KNN
  classifier over n-shot example files. Covers job creation, status
  polling, event log retrieval, output download, and config optimization.
  Do NOT use for real-time streaming Machine State (use newton-machine-state).
  Do NOT use for activity detection over video (use newton-activity-monitor).
  Do NOT use for file uploads (use newton-batch-upload).
---

# Newton Machine State Classification (Batch)

Create and monitor asynchronous batch jobs that classify time-series sensor data using the Machine State pipeline. Process large datasets (millions of rows) without maintaining a real-time session. The streaming counterpart is [`newton-machine-state`](../newton-machine-state/SKILL.md) — same encoder + classifier, different orchestration.

## When to Apply

- User wants to classify a large CSV dataset at scale
- User asks about batch processing or asynchronous inference
- User wants to run Newton on a file without a WebSocket session
- User needs to process multiple files through a pipeline
- User wants to optimize pipeline config for best accuracy

## Machine State Classification Pipeline

Classifies time-series sensor data using n-shot examples. Newton vectorizes each sensor window with an Omega encoder, then a hosted KNN classifier predicts the class label from the n-shot examples.

**Pipeline key:** `machine-state-classification`

This is the active deployment on both stage (`api.stage.u1.archetypeai.app`) and prod (`api.u1.archetypeai.app`) as of 2026-05-16 (`v1.1.1-409-e749ac0-rev5` on stage). The previously-documented `machine-state-job-pipeline` returns `Pipeline 'machine-state-job-pipeline' has no active versions` in both environments — it is not currently a usable key. The `machine-state-classification` deployment supports `omega_1_4_base` over arbitrary channel counts (verified with 4-channel vibration), so there is no channel-padding workaround required.

### Available model types

| Model | Channels | When to use |
|---|---|---|
| **`omega_1_4_base`** | **Arbitrary** (e.g. 4, 9, 52, 100+) | **Default — start here.** Generic encoder, no channel-count constraint. Suitable for almost every dataset (process plants, drilling, IoT, energy, …). Latest model. |
| `omega_1_3_surface` | Exactly 9 | Legacy domain-specific encoder for surface drilling sensors. Use only if you're working from an existing `omega_1_3_surface` n-shot library and don't want to re-shoot examples. |
| `omega_1_3_power_drive` | Exactly 9 | Legacy domain-specific encoder for downhole power-drive sensors. Same legacy guidance as above. |

For new projects pick `omega_1_4_base` — it removes the "exactly 9 sensors" constraint that made the 1.3 models painful to use on arbitrary plant/process data, and benchmarks at least as well across the domains we've tested.

### Input ports

- `worker.inference` — CSV files to classify
- `worker.n_shots` — labeled CSV example files with `metadata.class`

### Data requirements

- CSV files must include a timestamp column (Unix epoch or any monotonically increasing numeric).
- `data_columns` lists the sensor columns the encoder will see.
  - With `omega_1_4_base`, any positive count works.
  - With `omega_1_3_surface` / `omega_1_3_power_drive`, the count must be **exactly 9** — anything else throws a shape error (see *Common Errors*).
- `window_size` controls timesteps per classification window (typical: 16 / 32 / 64 / 128). Use `window_size=1` and you'll get a tensor error.
- N-shot files must produce at least `n_neighbors` windows: `(rows − window_size) / step_size ≥ n_neighbors`.
- For drilling specifically, the `omega_1_3_surface` n-shot labels should come from ACTC rig mode codes, not sensor heuristics — control-system ground truth is more reliable than thresholded readings.

### Example: process-plant classification with `omega_1_4_base` (52 variables)

Classifying a chemical-plant process dataset (the public [Tennessee Eastman Process](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1) benchmark — 52 process variables, 21 fault types) into binary `normal` vs `fault` from two n-shot example files. This is the canonical `omega_1_4_base` shape — the config below is what you should clone for any new project, regardless of domain.

```bash
curl -s -X POST "$BASE_URL/batch/jobs" \
  -H "Authorization: Bearer $ATAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "tep-classification",
    "pipeline_type": "batch",
    "pipeline_key": "machine-state-classification",
    "inputs": {
      "worker.inference": [{"file_id": "tep_inference.csv"}],
      "worker.n_shots": [
        {"file_id": "tep_normal.csv", "metadata": {"class": "normal"}},
        {"file_id": "tep_fault.csv",  "metadata": {"class": "fault"}}
      ]
    },
    "parameters": {
      "worker": {
        "parallelism": 1,
        "config": {
          "model_type": "omega_1_4_base",
          "batch_size": 32,
          "reader_config": {
            "data_columns": ["xmeas_1","xmeas_2", "...", "xmv_11"],
            "timestamp_column": "timestamp",
            "window_size": 64,
            "step_size": 1
          },
          "classifier_config": {
            "n_neighbors": 5,
            "metric": "euclidean",
            "weights": "uniform",
            "normalize_embeddings": false
          },
          "flush_every_n_iteration": 150
        }
      }
    }
  }'
```

Key knobs and how they differ from the 1.3 defaults:

- `model_type: "omega_1_4_base"` (was `omega_1_3_surface` / `omega_1_3_power_drive`).
- `batch_size: 32` (was 8) — 1.4 is faster per-window, larger encoder batches don't bottleneck.
- `flush_every_n_iteration: 150` (was 1000) — write outputs more often for better mid-job visibility.
- `classifier_config.normalize_embeddings: false` — new field exposed by the 1.4 pipeline. L2-normalizing the 768-dim embeddings before KNN occasionally helps when channel scales are wildly heterogeneous; default `false` matches what we've benchmarked best on so far.

### Example: drilling classification with the legacy 1.3 encoder

Pinned to `omega_1_3_surface` because the n-shot library exists for the 9 surface channels. Full source in [archetypeai-batch-examples-volve](https://github.com/archetypeai/archetypeai-batch-examples-volve).

```bash
curl -s -X POST "$BASE_URL/batch/jobs" \
  -H "Authorization: Bearer $ATAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "drilling-classification",
    "pipeline_type": "batch",
    "pipeline_key": "machine-state-classification",
    "inputs": {
      "worker.inference": [{"file_id": "volve_inference.csv"}],
      "worker.n_shots": [
        {"file_id": "volve_drilling.csv",     "metadata": {"class": "drilling"}},
        {"file_id": "volve_not_drilling.csv", "metadata": {"class": "not_drilling"}}
      ]
    },
    "parameters": {
      "worker": {
        "parallelism": 1,
        "config": {
          "model_type": "omega_1_3_surface",
          "batch_size": 8,
          "reader_config": {
            "data_columns": ["BPOS","DBTM","FLWI","HDTH","HKLD","ROP","RPM","SPPA","WOB"],
            "timestamp_column": "DATE_TIME",
            "window_size": 64,
            "step_size": 1
          },
          "classifier_config": {"n_neighbors": 5, "metric": "euclidean", "weights": "uniform"},
          "flush_every_n_iteration": 1000
        }
      }
    }
  }'
```

Drilling channel list (the required nine for `omega_1_3_surface`):
```
DATE_TIME, BPOS, DBTM, FLWI, HDTH, HKLD, ROP, RPM, SPPA, WOB
```

## Monitoring Jobs

### Check Status

```bash
curl -s "$BASE_URL/batch/jobs/$JOB_ID" \
  -H "Authorization: Bearer $ATAI_API_KEY"
```

Status lifecycle: `PENDING` → `RUNNING` → `COMPLETED` / `FAILED` / `CANCELLED`

### View Events/Logs

```bash
curl -s "$BASE_URL/batch/jobs/$JOB_ID/events" \
  -H "Authorization: Bearer $ATAI_API_KEY"
```

### Download Outputs

```bash
# List output artifacts (paginated, presigned S3 URLs, 1-hour expiry)
curl -s "$BASE_URL/batch/jobs/$JOB_ID/outputs?limit=50&offset=0" \
  -H "Authorization: Bearer $ATAI_API_KEY"

# Download artifact (no auth needed — signature is in the URL)
curl -s -o output.csv "$PRESIGNED_URL"
```

Machine State output format:
```csv
DATE_TIME,Prediction
1232522644,drilling
1190749684,not_drilling
```

### List All Jobs

```bash
curl -s "$BASE_URL/batch/jobs?limit=10&offset=0" \
  -H "Authorization: Bearer $ATAI_API_KEY"
```

## Config Optimization

The Machine State pipeline has several tunable hyperparameters. Use grid search with a small test dataset (200 rows) to find the best config before running on the full dataset:

| Parameter | Values to try | Description |
|-----------|--------------|-------------|
| `window_size` | 16, 32, 64, 128 | Time steps per classification window |
| `n_neighbors` | 3, 5, 7, 11 | KNN neighbors |
| `metric` | euclidean, cosine, manhattan | Distance metric |
| `weights` | uniform, distance | KNN weight function |

See [`optimize_config.py` in archetypeai-batch-examples-volve](https://github.com/archetypeai/archetypeai-batch-examples-volve/blob/main/3_batch_jobs/optimize_config.py) for a working grid-search script. The pattern transfers to `omega_1_4_base` unchanged — just swap `model_type` and `data_columns`.

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v0.5/batch/jobs` | POST | Create a batch job |
| `/v0.5/batch/jobs` | GET | List all jobs |
| `/v0.5/batch/jobs/{job_id}` | GET | Get job status |
| `/v0.5/batch/jobs/{job_id}/events` | GET | Get job events/logs |
| `/v0.5/batch/jobs/{job_id}/outputs` | GET | List output artifacts (paginated, presigned URLs) |

## Fine-Tuning (TBD)

Fine-tuning via `POST /v0.5/internal/experiment/runner/jobs` is not yet available. It requires JSONL-formatted training data — see the JSONL converter in [archetypeai-batch-examples-volve](https://github.com/archetypeai/archetypeai-batch-examples-volve).

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `shape '[-1, 6912]' is invalid for input of size N` | Using `omega_1_3_surface` / `omega_1_3_power_drive` with a `data_columns` count ≠ 9 | Either supply exactly 9 columns, or switch `model_type` to `omega_1_4_base` (no channel-count constraint) |
| `n_neighbors must be <= number of samples in index` | N-shot files too small for `window_size` | Increase n-shot rows or decrease `window_size` |
| `could not parse as dtype i64` | Mixed int/float values in CSV | Ensure all sensor values are formatted as floats |
| `Failed to resolve file inputs` | File not uploaded | Upload file first via Files API ([newton-batch-upload](../newton-batch-upload/SKILL.md)) |
| `Pipeline 'X' has no active versions` | Pipeline not deployed in this environment | Verify the pipeline_key is deployed — the active key on stage and prod is `machine-state-classification` (the newer `machine-state-job-pipeline` is documented but not currently deployed in either env; if a tool defaults to it you'll need to override). |

## Best Practices

- **Default to `omega_1_4_base`** — generic, no channel-count constraint, and the model we ship as the recommended starting point. Only fall back to `omega_1_3_*` if you have an existing n-shot library tied to that encoder.
- **Start with quick tests** — use 200-row samples (contiguous, not random — see TEP `tep_quick_test_200.csv`) to verify the pipeline before launching a multi-hour full run.
- **Use control-system labels for n-shots when available** — e.g. ACTC rig mode codes for drilling, plant DCS state codes for process plants. More reliable than thresholded-sensor heuristics.
- **Optimize config first** — grid search over hyperparameters with small data before committing to a multi-hour full run. Both example repos ship an `optimize_config.py`.
- **Poll status periodically** — jobs can run for minutes to hours depending on data size. Lower `flush_every_n_iteration` if you want mid-job progress visibility.
- **Check events on failure** — the `/events` endpoint provides detailed error messages including row-level context.
- **Watch n-shot window math** — `(rows − window_size) / step_size ≥ n_neighbors`. Easy to under-shoot when bumping `window_size` mid-experiment.

## Example Code

[**archetypeai-batch-examples-volve**](https://github.com/archetypeai/archetypeai-batch-examples-volve) — full public end-to-end pipeline for the Machine State batch flow: data prep → upload → batch job → outputs → evaluation → config optimization, in Python, shell, and curl. Pinned to `omega_1_3_surface` over 9 drilling channels (Volve North Sea well data), but the orchestration code, optimize_config grid search, and evaluation utilities transfer to `omega_1_4_base` unchanged — switch `model_type` and broaden `data_columns` to match your sensor set.

Use this repo as a template even if your default model is `omega_1_4_base`. The only Volve-specific bits are the `data_columns` list and the n-shot files; everything else (upload helpers, job creation, polling loop, output download, evaluation) is reusable as-is.
