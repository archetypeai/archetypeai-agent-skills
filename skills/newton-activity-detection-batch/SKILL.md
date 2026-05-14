---
name: newton-activity-detection-batch
description: >
  Run text-in / text-out asynchronous batch jobs on Archetype AI Newton
  via the `activity-detection` pipeline — the Newton C language model
  invoked over a JSONL prompt file. Use when the user wants to summarize,
  narrate, or reason over large structured datasets (CSV logs, sensor
  flows, event streams) at offline scale without holding a `/query`
  session open. Covers job creation, polling, output download, the
  empirical ~4K-token quality cliff, MapReduce / hierarchical reduce
  patterns for GB-scale inputs, N-way positional split for batch
  concurrency, and the silent join bugs (content-key vs. position-based
  join, `line_index` collisions after concat) that bite when chaining
  reduce stages.
  Do NOT use for time-series classification (use newton-machine-state-batch).
  Do NOT use for vision/video Q&A (use newton-activity-monitor).
  Do NOT use for synchronous text reasoning with a fixed-size state
  snapshot (use newton-query-prompting + the `/query` endpoint).
  Do NOT use for file uploads (use newton-batch-upload).
---

# Newton Activity Detection (Batch Text Generation)

Create and monitor asynchronous batch jobs that run the **Newton C language model** against a JSONL prompt file. Each line in the input is an `InferenceRecord` (`system` + `instruction` + `prompt` + optional `inputs` array); each line in the output is `{"line_index": N, "prediction": "..."}`. The pipeline is named `activity-detection` on the platform.

The streaming/synchronous counterpart is [`newton-query-prompting`](../newton-query-prompting/SKILL.md) — same C model, different orchestration. Use this skill when you need to fold thousands-to-millions of prompts asynchronously; use `/query` when you have one state snapshot and need an answer in 3–6 s.

Worked end-to-end example, including the full MapReduce code: [**archetypeai-batch-examples-ghost-iot**](https://github.com/archetypeai/archetypeai-batch-examples-ghost-iot) — 1 GB of synthetic WiFi flow CSV (5.8 M flows across 24 devices × 24 hours) folded into 9 daily narratives via six reduce stages. The numbers and pattern depths quoted below come from that pipeline; the patterns are model- and dataset-independent — only the depth of the hierarchy changes with the cliff value.

## When to Apply

- User wants Newton to write per-row / per-record narratives over a large dataset (logs, flows, events) where the input is too big or too numerous for `/query`.
- User wants to fold many small inputs into one summary via MapReduce / hierarchical reduce.
- User is hitting silent quality degradation on long prompts and isn't sure why ("status COMPLETED, predictions are garbage").
- User is chaining reduce stages and getting cross-scope leakage in the final narratives.

## Pipeline shape

```
Pipeline key:   activity-detection
Input port:     worker.data        (one JSONL file; each line = InferenceRecord)
Output:         outputs/pred_*.jsonl
                  one JSON per line: {"line_index": N, "prediction": "..."}
```

### Input record schema

```json
{
  "system": "You are a network security analyst reviewing smart-home WiFi traffic...",
  "instruction": "Analyze the attached flow log and describe what happened on this day. Summarize activity level, dominant protocols, temporal patterns, most-active device, and flag anything unusual.",
  "prompt": "Date: 2019-10-19 UTC. Scope: home wlan0 interface. Flow count: 78.",
  "inputs": [
    {
      "type": "text",
      "format": "plain",
      "data": "Flow log fields (pipe-separated): time_utc|mac_a|mac_b|prot|tran|port_a|port_b|bytes_a|bytes_b|pkts_a|pkts_b. Transport: 6=TCP, 17=UDP, 1=ICMP.\n\n15:55:11|ebd1a7fa8544|e323b826aa71|DHCP|17|67|68|0|900|0|3\n15:55:14|e323b826aa71|13d35af5c06b|DHCP|17|68|67|900|0|3|0\n..."
    }
  ]
}
```

The **question** goes in `prompt`, the **evidence** goes in `inputs[0].data`. The `inputs` array is the InferenceRecord schema's contract for extra context (text/image/video). Full schema: [Nano Inference input format](https://github.com/archetypeai/atai_core/tree/main/services/jos_service/nano_inference#input-format).

## Creating a Batch Job

```bash
curl -s -X POST "$BASE_URL/batch/jobs" \
  -H "Authorization: Bearer $ATAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-narrative-job",
    "pipeline_type": "batch",
    "pipeline_key": "activity-detection",
    "inputs": {
      "worker.data": [{"file_id": "my_prompts.jsonl"}]
    },
    "parameters": {
      "worker": {
        "parallelism": 1,
        "config": {
          "generation": {
            "do_sample": true,
            "max_new_tokens": 1024,
            "repetition_penalty": 1,
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.8
          }
        }
      }
    }
  }'
```

Response: `{"id": "job_...", "status": "PENDING", ...}`. The standard lifecycle is `PENDING → RUNNING → COMPLETED / FAILED / CANCELLED`.

### Generation knobs

| Knob | Typical | Notes |
|---|---|---|
| `max_new_tokens` | 512–1024 | Newton truncates mid-sentence at the cap. Size for your longest expected narrative + ~20 % headroom. |
| `temperature` | 0.5–0.8 | Lower = more consistent across reduce passes (useful when you'll concatenate predictions back). |
| `do_sample` | `true` | `false` gives greedy decoding — sometimes useful for deterministic fold stages. |
| `top_k` / `top_p` | 20 / 0.8 | Defaults match `/query`. Rarely worth tuning unless you see repetition. |

## Monitoring & Outputs

```bash
# Poll status
curl -s "$BASE_URL/batch/jobs/$JOB_ID" -H "Authorization: Bearer $ATAI_API_KEY"

# Events / logs
curl -s "$BASE_URL/batch/jobs/$JOB_ID/events" -H "Authorization: Bearer $ATAI_API_KEY"

# List output artifacts (paginated, presigned S3 URLs, 1-hour expiry)
curl -s "$BASE_URL/batch/jobs/$JOB_ID/outputs?limit=50&offset=0" \
  -H "Authorization: Bearer $ATAI_API_KEY"

# Download an artifact (no auth — signature is in the URL)
curl -s -o pred_0.jsonl "$PRESIGNED_URL"
```

Output line:

```json
{"line_index": 0, "prediction": "On 2019-10-19 the home wlan0 interface saw 78 flows..."}
```

On per-record failure:

```json
{"line_index": 5, "prediction": null, "error": "parse error"}
```

`"parse error"` almost always means **the input wasn't JSONL** (raw CSV submitted by mistake, or a stray blank line). The job itself still returns `COMPLETED`.

## The Quality Cliff (read this before designing anything)

The Newton C model has a **nominal context budget of 16,384 tokens** but a much tighter **practical quality cliff at ~4K tokens (~16 KB of pipe-separated structured rows)**. Above the cliff the model silently degrades into table-completion mode and returns 1–3-character fragments like `'00'`, `'|153|'`, or empty strings.

**The failure is silent.** No `WARN` event, no `FAILED` status. The job reports `Status: COMPLETED`, `0 failed`. The only signal is reading the predictions themselves.

Empirical cliff sweep on CSV-heavy `inputs[0].data` (from the [ghost-iot worked example](https://github.com/archetypeai/archetypeai-batch-examples-ghost-iot), §10.6 of the README):

| `inputs[0].data` size | ~Tokens | Output |
|---|---|---|
| 10.4 KB | ~2.5K | Real narratives ✓ |
| **16.5 KB** | **~4K** | **Real narratives ✓ (last good)** |
| 20.6 KB | ~5K | ✗ Garbage fragments |
| 41.1 KB | ~10K | ✗ Garbage + truncation warning fires |

The cliff and the nominal context window are **independent failure modes**. The model can ingest the tokens and emit plausible-sized output — it just can't synthesize narrative from that much structured data at once.

### Cliff sweep methodology

Before designing a multi-day pipeline, spend 30 minutes establishing the cliff for *your* model + *your* input shape:

1. Take 5 representative records.
2. Build 4 variants per record, capped at 10 KB / 16 KB / 20 KB / 40 KB of `inputs[0].data`.
3. Submit all 20 as a single batch job.
4. Read every prediction. The cliff is the largest cap where all 5 still produce real narratives.

**Revalidate per model and per input shape.** A model with a larger nominal context window doesn't automatically have a higher cliff — CSV-heavy structured input degrades earlier than prose. Newton 2.5.1, Gemma, Claude, GPT-4o will all have different cliffs.

## MapReduce / Hierarchical Reduce

Once you know the cliff, every reduce stage in a pipeline answers the same question:

> Does the concatenation of N inputs at this stage fit under the cliff?

- **Yes** → single-pass reduce works.
- **No** → insert a hierarchical pre-fold (group N inputs into packs of K, reduce each pack into a super-input, then fold super-inputs).

### Worked shape — folding 1 GB of CSV into 9 narratives

From the [ghost-iot worked example](https://github.com/archetypeai/archetypeai-batch-examples-ghost-iot), §8 of the README — observed across one end-to-end run on the C 2.4 model (1 GB of WiFi flow CSV, 5.8 M flows across 24 devices × 24 hours, multi-tenant topology of 3 homes × 2 humans × 4 devices/human):

```
chunks (≤16 KB each, every flow preserved) → bucket reduce
  ├─ obvious: fold all chunks of a (device, hour) bucket → ✗ overflows for ~all buckets
  └─ hierarchical:
       Stage A:   chunks → partials       (groups of K=3 chunks per partial)
       Stage A₂:  partials → super-partials (groups of K=4 partials per super) — needed because Stage A's
                                              busy buckets had up to 44 partials = ~44 KB concat
       Stage B:   super-partials → 1 narrative per (device, hour) ✓

hourly narratives → device-day reduce
  ├─ obvious: fold 24 hours per device → ✗ overflows for the busiest device (17.8 KB)
  └─ hierarchical:
       Stage A:   hours → super-hours (groups of K=4 hours per super-hour)
       Stage B:   super-hours → 1 narrative per device ✓

device-day narratives → user-day / house-day reduce
  └─ single-pass:  3 / 8 inputs per group, fits → ✓
```

Six reduce passes total, folding 23,249 chunks down to 9 final narratives. The hierarchical layers exist because the cliff is ~16 KB; **a model with a higher cliff lets you collapse layers**, not change the pattern.

### The shape, restated

```python
# Pseudo-code for one hierarchical reduce pass.
# Inputs: list of N narratives (from a prior stage)
# Output: list of ceil(N/K) super-narratives

# Stage A: pack the inputs into groups of K and submit as a batch job
groups = [inputs[i:i+K] for i in range(0, len(inputs), K)]
for group_index, group in enumerate(groups):
    record = {
        "system":      SYSTEM_PROMPT,
        "instruction": "Summarize the following K narratives into one super-summary.",
        "prompt":      f"super-group {group_index}, K={len(group)}",
        "inputs":      [{"type": "text", "format": "plain",
                         "data":  "\n\n---\n\n".join(group)}],
    }
    emit_jsonl(record)

# Validate the cliff: max(len(record["inputs"][0]["data"])) must be < CLIFF_BYTES
# If it isn't, decrease K and re-pack.
```

Pick `K` so the worst-case packed size sits comfortably below the cliff. The ghost-iot pipeline settled at `K=3` for chunks→partials and `K=4` for the higher layers.

## N-way Batch Split (concurrency, optional)

Most orgs have batch concurrency ≥ 1. If yours has N > 1 slots, every stage can run N batch jobs in parallel, cutting wall-clock by ~N× **with no quality cost**. The recipe:

1. Build the per-stage JSONL as a single ordered file (`combined.jsonl`). **Do not shuffle.**
2. Split it positionally into N contiguous halves: `combined_p1.jsonl`, ..., `combined_pN.jsonl`.
3. Upload all N (see [`newton-batch-upload`](../newton-batch-upload/SKILL.md) if any half exceeds 255 MB).
4. Submit N batch jobs in parallel, one per half.
5. After all complete, concatenate the prediction files **in the same positional order** and renumber `line_index` sequentially (see next section).

Skip the split if N=1 (single-slot org). Use whatever N your org allows — the pattern is the same.

**Why positional, not random:** the join from predictions back to the source manifest depends on either index or content. Random shuffle destroys index alignment and complicates content-key reconstruction (see §"Two silent join bugs" below).

## Two silent join bugs when chaining reduce stages

Both bugs produce zero errors and zero warnings — `COMPLETED` status, plausible-looking but wrong narratives. **Always inspect at least one final narrative against its manifest scope** before trusting the pipeline.

### Bug 1 — Position-based join after shuffle

If you `shuf` the master JSONL before splitting, each output prediction's `line_index` is relative to the *shuffled* input, not the original manifest. A position-based join `predictions[i] → manifest[i]` quietly maps every prediction to the wrong scope (wrong device, wrong hour, wrong home).

**Symptom:** cross-scope leakage in final narratives — Home A's narrative cites devices from Home B.

**Fix — content-key join.** For every reduce stage that needs to know what each prediction is *about*, encode the identifying tuple inside the prompt text and extract it back from the prediction record:

```python
# At prep time, embed the key in the prompt:
prompt = (f"Scope: device={device_id}, hour={hour}, n_flows={n_flows}, "
          f"ts_lo={ts_lo}, ts_hi={ts_hi}, n_bytes={n_bytes}")

# After predictions arrive, extract the key from the original prompt text
# and join to the manifest by (device_id, hour, n_flows, ts_lo, ts_hi, n_bytes).
# This is bulletproof under shuffle, split, retry, or any reordering.
```

A content key is essentially free to compute and survives every reordering. Use it.

### Bug 2 — `line_index` collisions after concat-of-splits

Each split batch job emits predictions with `line_index` relative to *its own* input file (`0..N/2-1` for a half-split). Cat'ing the halves produces **duplicate** `line_index` values.

If any downstream prep script loads predictions into a `dict[line_index] → prediction` and looks up by `line_index`, dict insertion order means the last-loaded value wins:

- Manifest entries with `line_index 0..N/2-1` → silently get the *second half's* predictions (wrong scope).
- Manifest entries with `line_index N/2..N-1` → silently get **no prediction at all** (empty string).

**Symptom:** the next reduce stage describes "no activity" for scopes that should be busy.

**Fix — renumber `line_index` sequentially to `0..N-1` after every concat**, before passing predictions to the next stage's prep script. Five lines of Python:

```python
import json, sys
for new_idx, line in enumerate(sys.stdin):
    rec = json.loads(line)
    rec["line_index"] = new_idx
    print(json.dumps(rec))
```

Run this after every `cat split_*.jsonl`. Skipping it silently breaks the next reduce stage.

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/v0.5/batch/jobs` | POST | Create a batch job |
| `/v0.5/batch/jobs` | GET | List all jobs |
| `/v0.5/batch/jobs/{job_id}` | GET | Get job status |
| `/v0.5/batch/jobs/{job_id}/events` | GET | Get job events / logs |
| `/v0.5/batch/jobs/{job_id}/outputs` | GET | List output artifacts (paginated presigned URLs) |

## Common Errors

| Error | Cause | Fix |
|---|---|---|
| Every prediction is `null` with `"error": "parse error"` | Input is not JSONL (raw CSV, blank lines, trailing comma) | Re-emit the file with one `json.dumps(record)` per line and no blank lines |
| `COMPLETED`, predictions are 1–3-char fragments like `'00'`, `'|153|'`, `''` | `inputs[0].data` is above the model's quality cliff (~4K tokens / ~16 KB for Newton C 2.4) | Run a cliff sweep, then insert a hierarchical pre-fold so every record stays under the cliff |
| `COMPLETED`, final narratives reference the wrong scope | Position-based join after shuffle/split (Bug 1) | Switch to content-key join (encode identifying tuple in `prompt`) |
| `COMPLETED`, half the manifest reports "no activity" | `line_index` collisions after concat-of-splits (Bug 2) | Renumber `line_index` sequentially after every concat |
| `Failed to resolve file inputs` | File not uploaded | Upload first via [`newton-batch-upload`](../newton-batch-upload/SKILL.md) |
| `Pipeline 'X' has no active versions` | Pipeline key wrong for this environment | Verify `pipeline_key` is `activity-detection` and that your org has it deployed |

## Best Practices

- **Run a cliff sweep first.** 30 minutes of testing 5 records × 4 size caps pays for itself many times over — the whole architecture follows from that single number.
- **Default to hierarchical reduce when N inputs × per-input size approaches the cliff.** Same shape at every level (group K → super → fold).
- **Content-key joins, always.** Position-based joins look fine until the first shuffle or retry and then break silently. Encode `(scope_id, count, time_range, ...)` in `prompt` and parse it back.
- **Renumber `line_index` after every concat.** Five-line Python loop. Skipping it silently breaks the next stage.
- **Inspect at least one final narrative against its manifest scope.** Events log is not enough — both silent failure modes (cliff, joins) produce `COMPLETED` with bad content.
- **Use lower `temperature` (0.5–0.6) for intermediate fold stages.** Variance compounds across reduce passes; tighter sampling at the bottom keeps the final narrative coherent.
- **Split positionally for concurrency, never randomly.** Random shuffle complicates content-key reconstruction and gains nothing.
- **The hierarchy depth follows from the cliff, not the nominal context window.** A model with a higher cliff lets you collapse layers; same pattern, less depth.

## Example Code

[**archetypeai-batch-examples-ghost-iot**](https://github.com/archetypeai/archetypeai-batch-examples-ghost-iot) — full public end-to-end pipeline for the `activity-detection` batch flow: data prep → upload → batch jobs → outputs → join → view, in Python, shell, and curl. Covers both the **simple single-record pattern** (§2–6 of the README — one prompt summarizing a whole filtered day) and the **multi-home MapReduce pattern at GB scale** (§8 — 23 K chunks → six reduce stages → 9 daily narratives). Has the cliff-sweep methodology (§8.2 / §10.6), content-key join (§10.7), `line_index` renumbering (§10.8), and observed wall-clock across all stages.

The orchestration code is pipeline-agnostic — `upload_multipart.py`, the polling loop, and the paginated outputs download transfer unchanged to any other `pipeline_key`. Only the prep scripts (which build the per-stage JSONL) are activity-detection-specific.

See also [`EXPERIENCE.md`](https://github.com/archetypeai/archetypeai-batch-examples-ghost-iot/blob/main/EXPERIENCE.md) in that repo — long-form writeup of *why* the design ended up needing six MapReduce stages instead of one fold.
