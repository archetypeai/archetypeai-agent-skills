---
name: newton-machine-state
description: >
  Build n-shot classification pipelines using Newton's Machine State Lens.
  Use when the user wants to classify sensor data into categories (e.g.,
  stressed vs. relaxed, normal vs. anomaly, idle vs. active), perform
  anomaly detection on time-series data, or implement n-shot learning
  with CSV sensor data. This skill covers session lifecycle, focus CSV
  uploads, sliding window configuration, and SSE result parsing.
  Do NOT use for vision-based analysis or chart reading (use newton-activity-monitor).
  Do NOT use for initial API setup (use newton-setup).
---

# Newton Machine State Lens

Classify time-series sensor data using n-shot learning. The Machine State Lens uses labeled CSV examples (focus files) to classify new data streams via KNN over sliding windows.

## When to Apply

- User wants to classify sensor data into discrete states
- User asks about anomaly detection or state monitoring
- User wants n-shot classification without training a model
- User is building a real-time monitoring dashboard with status indicators
- User references HRV stress detection or OBD2 vehicle health patterns

## Lens ID

```
lns-1d519091822706e2-bc108andqxf8b4os
```

## Core Concepts

### N-Shot Classification

Provide labeled CSV examples (focus files) — one per class — and Newton classifies new data against those examples. No model training required.

### Sliding Windows

Newton processes CSV data in overlapping windows:
- **window_size**: Number of rows per window (default: 16)
- **step_size**: Rows to advance between windows (default: 8)
- Expected windows = `floor((dataPoints - windowSize) / stepSize) + 1`

## Workflow

### Step 1: Prepare Focus CSVs

Create one CSV per class with labeled sensor data. Each CSV should have:
- Consistent column names matching your data stream
- A representative sample of that class (50-200 rows recommended)
- No header row mismatches between focus files and query data

**Example: HRV Stress Detection**
```
# focus_relaxed.csv
rmssd,sdnn,mean_hr,pnn50,sd1
42.5,45.2,68.3,22.1,30.1
...

# focus_stressed.csv
rmssd,sdnn,mean_hr,pnn50,sd1
18.2,22.1,95.7,8.3,12.9
...
```

**Example: Vehicle Health Monitoring**
```
# focus_normal.csv
rpm,speed,coolant_temp,iat,engine_load,throttle,map
800,0,90,35,18,12,34
...

# focus_attention.csv
rpm,speed,coolant_temp,iat,engine_load,throttle,map
850,0,108,52,45,18,42
...
```

### Step 2: Upload Focus Files (One-Time)

```typescript
async function uploadFile(filePath: string, content: string): Promise<string> {
  const formData = new FormData();
  formData.append("file", new Blob([content], { type: "text/csv" }), filePath);

  const response = await newtonFetch("/files", {
    method: "POST",
    body: formData,
  });
  const data = await response.json();
  return data.file_id;
}
```

Cache the returned `file_id` values — focus files only need to be uploaded once per session.

### Step 3: Create a Session

```typescript
const response = await newtonFetch("/lens/sessions/create", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ lens_id: "lns-1d519091822706e2-bc108andqxf8b4os" }),
});
const { session_id } = await response.json();
```

### Step 4: Configure Session with Focus Files

Send a `session.modify` event with n-shot focus file IDs and CSV config:

```typescript
await newtonFetch("/lens/sessions/events/process", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    session_id,
    events: [{
      kind: "session.modify",
      payload: {
        input_n_shot: [
          { label: "relaxed", file_id: relaxedFileId },
          { label: "stressed", file_id: stressedFileId },
        ],
        csv_configs: {
          window_size: 16,
          step_size: 8,
        },
      },
    }],
  }),
});
```

### Step 5: Connect SSE Consumer (Before Input!)

**Critical:** Always connect the SSE consumer before setting the input stream to avoid missing results.

```typescript
const sseResponse = await newtonFetch(
  `/lens/sessions/consumer/${sessionId}`,
  { headers: { Accept: "text/event-stream" } }
);
```

### Step 6: Set Input Stream

Upload query CSV and set as input:

```typescript
const queryFileId = await uploadFile("query.csv", csvContent);

await newtonFetch("/lens/sessions/events/process", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    session_id: sessionId,
    events: [{
      kind: "input_stream.set",
      payload: {
        streams: [{
          id: "input-0",
          source: { type: "csv_file_reader", file_id: queryFileId },
        }],
      },
    }],
  }),
});
```

### Step 7: Parse SSE Results

Each `inference.result` event contains a classification for one window:

```typescript
// SSE event data shape
{
  "kind": "inference.result",
  "payload": {
    "result": ["stressed", { "stressed": 0.85, "relaxed": 0.15 }]
  }
}
```

Aggregate across windows — majority vote or average scores:

```typescript
function aggregateResults(results: Array<[string, Record<string, number>]>) {
  const totals: Record<string, number> = {};
  for (const [label, scores] of results) {
    for (const [key, score] of Object.entries(scores)) {
      totals[key] = (totals[key] || 0) + score;
    }
  }
  const count = results.length;
  return Object.fromEntries(
    Object.entries(totals).map(([k, v]) => [k, v / count])
  );
}
```

### Step 8: Early SSE Termination (Optimization)

Calculate expected window count and close the stream early instead of waiting for the idle timeout (60-80s):

```typescript
const expectedWindows = Math.floor((dataPoints - windowSize) / stepSize) + 1;
let receivedWindows = 0;

// In SSE parse loop:
if (++receivedWindows >= expectedWindows) {
  reader.cancel(); // Close early
}
```

## Session Lifecycle

```
Upload Focus CSVs (once) → Create Session (once) → [Query Loop: Upload CSV → Set Input → Read SSE Results] → Destroy Session
```

Sessions are reusable — create once, query many times. Only recreate on error.

### Cleanup

```typescript
// Destroy session
await newtonFetch("/lens/sessions/destroy", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ session_id: sessionId }),
});

// Delete uploaded files
await newtonFetch(`/files/delete/${fileId}`, { method: "DELETE" });
```

## NewtonStreamManager Singleton Pattern

For real-time applications, implement a server-side singleton that:

1. Buffers incoming sensor data (max ~300 points)
2. Runs periodic queries every 15 seconds
3. Manages session lifecycle (create on first query, reuse, recreate on error)
4. Pushes results to frontend via SSE

See [references/stream-manager-pattern.md](references/stream-manager-pattern.md) for the full implementation pattern.

## Parallel Per-Subsystem Sessions

A single shared session produces one verdict for the whole system. If you need to know *which* subsystem saw the anomaly — to route alerts, rank severity, or drive per-component UI — run **N parallel sessions, one per subsystem, each over a column subset of the same wide CSV**.

Canonical shape: the subsystems share the same n-shot focus files (`swat_normal.csv` / `swat_attack.csv`) but each lens registers with a different `data_columns` filter — P1's session sees `FIT101/LIT101/MV101/P101`, P3's session sees its nine UF columns, and so on. Six classifiers, one problem, six per-subsystem verdicts per window.

```js
// Per-stage lens register — same n-shot files, different column subset
{
  lens_name: `swat-stage-${stageId}-${ts}`,
  model_pipeline: [{ processor_name: 'lens_timeseries_state_processor' }],
  model_parameters: {
    model_name: 'OmegaEncoder',
    model_version: 'OmegaEncoder::omega_embeddings_01',
    buffer_size: 30,
    input_n_shot: { NORMAL: normalFileId, ATTACK: attackFileId },
    csv_configs: {
      data_columns: STAGE_COLUMNS[stageId],  // <-- only this stage's sensors
      window_size: 30,
      step_size: 30
    },
    knn_configs: { n_neighbors: 3, metric: 'euclidean' }
  }
}
```

Streaming per window fans out too — transpose the current window to channel-first (`[[col1 values], [col2 values], ...]`, not row-major) and `POST /lens/sessions/events/process` to each session in parallel. On the consumer side the browser opens N concurrent `EventSource` connections (one per session) and bucket-sorts incoming `inference.result` events by session ID to update per-subsystem state.

**When to prefer N parallel sessions over 1 shared session:**
- You need component-level verdicts (to color a plant schematic, page a specific oncall rotation, surface per-area suggestions)
- Sensors are naturally grouped by subsystem, with limited cross-group interaction
- 6× the session-create calls at startup is acceptable (`Promise.all` makes it ~1.5s instead of 9s sequential)

**When the singleton pattern is better:**
- You only care about a single "is the system OK" verdict
- Subsystems are tightly coupled and the anomaly is usually observable from any sensor
- Session count matters for cost (most Newton pricing bills by inference count, not session count — but verify for your account)

See [references/parallel-subsystem-pattern.md](references/parallel-subsystem-pattern.md) for the full pattern, including browser-side cleanup on tab close.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 16 | Rows per sliding window |
| `step_size` | 8 | Row stride between windows |
| `MAX_BUFFER_SIZE` | 300 | Max sensor readings to buffer |
| `MIN_DATA_POINTS` | 20-32 | Minimum points before first query |
| `QUERY_INTERVAL` | 15000ms | Time between periodic queries |

## Best Practices

- **Focus CSV quality matters more than quantity** — 50-200 well-labeled rows per class is sufficient
- **Column names must match** between focus files and query data exactly
- **Reuse sessions** — creating sessions has overhead; create once and query repeatedly
- **Always connect SSE before setting input** — otherwise results may be missed
- **Implement early termination** — avoids 60-80s idle wait per query
- **Graceful degradation** — apps should work without Newton; check availability first
- **Clean up on tab close** (browser-side sessions) — fire `DELETE /lens/sessions/destroy` from a `pagehide` handler with `fetch(..., { keepalive: true })` so orphaned sessions don't accumulate on refresh/close. `navigator.sendBeacon` works too but only sends POSTs.
- **Clean stale lenses on startup** — before registering a new lens, `GET /lens/metadata` and delete any old ones matching your name prefix. Lens registrations persist across sessions and accumulate.
- **Channel-first transpose** — Machine State Lens expects streams as `[[col1 values], [col2 values], ...]`, not row-major. If classifications look random, check this first.

## Building a frontend on top of this

If you're wrapping the Machine State Lens in a React/Svelte/etc. UI — per-stage anomaly dashboard, real-time classification monitor, n-shot replay tool — **read [`DESIGN.md`](../../DESIGN.md) at the root of this repo before writing any CSS**. The Archetype design system (Tailwind v4 + `@archetypeai/ds-lib-tokens` + PP Neue Montreal sans/mono + OKLCH palette + dark-first) is the expected visual language for these demos. [`newton-swat-demo`](https://github.com/archetypeai/newton-swat-demo) is the canonical reference implementation for Machine State (6 parallel SSE sessions, per-stage cards with `good` / `warning` / `critical` Badge variants, mono numeric readouts, sharp 2px radii); [`newton-wifi-demo`](https://github.com/archetypeai/newton-wifi-demo) shows the same patterns applied to a different domain. Setting this up at the start is much cheaper than retrofitting later.
