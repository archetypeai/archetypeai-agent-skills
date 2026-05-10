---
name: newton-query-prompting
description: >
  Prompt-engineering patterns for Newton's /query text-reasoning endpoint.
  Use when the user wants Newton to reason over structured state and return
  JSON/structured output — e.g. operator suggestions from sensor readings,
  categorized findings from log data, routing decisions from event streams.
  Covers structured-output enforcement, contamination-avoidance, constraint
  tables, server-side pre-picking of salient inputs, and topology validation
  on parsed responses.
  Do NOT use for classification of raw sensor data (use newton-machine-state).
  Do NOT use for vision/video analysis (use newton-activity-monitor).
  Do NOT use for initial API setup (use newton-setup).
---

# Newton /query Prompt Engineering

Newton's `/query` endpoint is a text-reasoning path: the model sees a system prompt, a query (your current state snapshot), and returns a text response. Unlike the Lens APIs, there's no fixed output schema — the prompt is the schema. This skill covers the prompt patterns that make `/query` reliable in production.

Reference implementation: the Suggested Actions panel in [newton-swat-demo](https://github.com/archetypeai/newton-swat-demo) uses these patterns end-to-end. See also [docs/suggested-actions-prompt.md](https://github.com/archetypeai/newton-swat-demo/blob/main/docs/suggested-actions-prompt.md) for a full worked example.

## When to Apply

- User wants Newton to output structured JSON (arrays, objects, typed fields) rather than prose
- User needs Newton to route outputs by topology (source→target, categories, zones)
- User observes Newton picking familiar-looking identifiers over genuinely salient ones
- User observes Newton copying example values verbatim instead of substituting its own
- User observes Newton returning prose/markdown when they asked for JSON
- User's current prompt works sometimes but fails under specific states

## Endpoint

```
POST {ATAI_API_ENDPOINT}/v0.5/query
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

Latency is typically **3–6s p50, ~1–2s warmup on first call**. Can be called directly from the browser if the server proxy adds latency (see "Direct browser calls" below).

## Request Body

```json
{
  "query": "<per-request state snapshot>",
  "system_prompt": "<SYSTEM_PROMPT>",
  "instruction_prompt": "<same as system_prompt>",
  "file_ids": [],
  "model": "Newton::c2_4_7b_251215a172f6d7",
  "max_new_tokens": 700,
  "sanitize": false
}
```

Notes on each parameter:

- **`model`** — current Newton checkpoint ID for your account. Check `/v0.5/models` if unsure.
- **`max_new_tokens`** — size for your expected output, plus ~20% headroom. Newton truncates mid-sentence when it runs out of budget. Observed sane defaults: 300 for short structured output, 700 for a dozen JSON cards, 1500+ for long-form prose.
- **`sanitize: false`** — **required for structured output**. `sanitize: true` rewrites punctuation (smart quotes, dashes) and breaks `JSON.parse` downstream.
- **`system_prompt` + `instruction_prompt`** — Newton's chat template uses `instruction_prompt` as the authoritative system turn; `system_prompt` is a legacy alias. Send the same string to both — cheapest insurance.
- **`file_ids: []`** — no retrieval; the prompt is fully self-contained. Populate only if you've uploaded reference docs for RAG-style context.

## The Five Prompt Patterns

### 1. Structured-output enforcement

Newton *will* wrap responses in markdown fences (` ```json ... ``` `), add preambles ("Here are the actions you should take:"), and sometimes explain itself afterwards. Three mitigations, in order of leverage:

**(a) Say it loudly in the system prompt:**

```
Return ONLY a JSON array. No prose, no markdown code fences, no explanation
— just the JSON. Shape: [{"field":"..."}]
```

**(b) Specify the exact shape with a type-like signature:**

```
Shape: [{"origin":"Pn","target":"Pm","direction":"upstream|local|downstream","text":"..."}]
```

Enumerated string unions (`"upstream|local|downstream"`) are honored reliably; free-form strings get creative.

**(c) Parse defensively.** Always strip fences and extract the outermost JSON before parsing:

```js
function parseStructured(text) {
  const cleaned = text
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```\s*$/i, '')
    .trim();
  const start = cleaned.indexOf('[');
  const end = cleaned.lastIndexOf(']');
  if (start === -1 || end === -1 || end <= start) return null;
  try {
    return JSON.parse(cleaned.slice(start, end + 1));
  } catch {
    return null;
  }
}
```

### 2. Constraint tables for routing / topology

If outputs need to map to a fixed graph (stage → neighbor, category → handler, zone → oncall), inline the full mapping in the system prompt as a table and repeat the rule:

```
TOPOLOGY — the "target" field MUST EXACTLY match these pairings.
Never substitute or reassign:
- P1 anomaly: upstream=none, local=P1, downstream=P2
- P2 anomaly: upstream=P1, local=P2, downstream=P3
- P3 anomaly: upstream=P2, local=P3, downstream=P4
- P4 anomaly: upstream=P3, local=P4, downstream=P5
- P5 anomaly: upstream=P4, local=P5, downstream=P6
- P6 anomaly: upstream=P5, local=P6, downstream=none

If a direction maps to "none" for a given origin, DO NOT emit that direction.
```

Then **validate server-side** after parsing. Newton occasionally routes cards wrong (e.g. P3 upstream → P1 instead of P2) no matter how forcefully you state the rule. Drop violators:

```js
const TOPOLOGY = {
  P1: { upstream: null, local: 'P1', downstream: 'P2' },
  /* ... */
};

const valid = parsed.filter((s) => {
  const topo = TOPOLOGY[s.origin];
  if (!topo) return false;
  const expected = topo[s.direction];
  if (expected === null) return false;
  return s.target === expected;
});
```

### 3. Avoid example contamination

Concrete identifiers in your prompt get copied verbatim by Newton. If your example says:

```
Example: {"sensor":"FIT101","value":0.00,"action":"check valve MV101"}
```

...Newton will cite `FIT101` and suggest checking `MV101` even when the actual anomaly is on `AIT402`. Two fixes:

**(a) Use placeholder names for shape examples:**

```
Example shape (use sensor name from state, not this placeholder):
{"sensor":"ZZZ000","value":0.00,"action":"check valve"}
```

**(b) Use generic Pn/XXn placeholders for category examples:**

```
{"origin":"PX","target":"PY","direction":"upstream",
 "text":"<sensor>=<value> — reduce feed from PY"}
```

Newton recognizes these as structural placeholders and doesn't paste them into output.

### 4. Pre-pick salient inputs server-side

Newton has strong priors toward familiar-looking identifiers. If your state has 40 sensors and you hand them all over with "pick the most anomalous," Newton will often cite whichever one has the most common-looking name (e.g., `FIT101`, `LIT101`) rather than the one actually deviating.

Pre-pick server-side. Rank by whatever signal defines "salient" in your domain (z-score against baseline, rate of change, rule score) and give Newton a single choice per group with an emphatic "cite this" instruction:

```js
function pickTopDeviation(sensors, baselines) {
  let best = null;
  for (const [col, val] of Object.entries(sensors)) {
    const b = baselines[col];
    if (!b) continue;
    const z = Math.abs((val - b.mean) / (b.std || 0.0001));
    if (!best || z > best.z) best = { col, val, z };
  }
  return best;
}

// In query body:
const top = pickTopDeviation(stageSensors, baselines);
line += `\n    cite this sensor: ${top.col}=${top.val.toFixed(2)} ` +
        `z=${top.z.toFixed(1)} (strong deviation)`;
```

And in the system prompt:

```
CRITICAL: use the sensor name+value specified after "cite this sensor:"
for that stage. Do NOT substitute a different sensor even if another
looks more familiar.
```

This single pattern was the biggest quality win in the SWaT demo — from "Newton cites FIT101 regardless of actual anomaly" to "Newton cites the top-z sensor every time."

### 5. Multi-part output with explicit separators

If each output item needs two or more distinct sub-fields (e.g., `citation + action`, `finding + severity + owner`), specify a separator and both halves:

```
Each "text" field is a full instruction with TWO parts joined by " — ":
  part 1: the EXACT sensor citation from that stage's "cite this sensor:" line
          (copy sensor name and value verbatim; drop the z annotation)
  part 2: an imperative operator action with a concrete verb
          (reduce, check, isolate, alert, hold, bypass)
```

Without the separator, Newton picks one half and drops the other. Concrete verb lists keep the "action" half from devolving into passive-voice observations.

## Response Shape

Newton's response body:

```json
{ "response": { "response": ["<string>"] } }
```

The string is your raw output. Defensive unwrapping (shape varies across model versions):

```js
let raw = '';
if (data.response?.response && Array.isArray(data.response.response)) {
  raw = data.response.response[0] || '';
} else if (Array.isArray(data.response)) {
  raw = data.response[0] || '';
} else if (typeof data.response === 'string') {
  raw = data.response;
} else if (data.text) {
  raw = data.text;
}
```

## Direct Browser Calls

If your server proxy adds serialization overhead or wedges on long-running `/query` calls, call Newton directly from the browser. Steps:

1. Ship an API key endpoint (`/api/session/one`) that returns a per-client short-lived key + endpoint URL, so the real key isn't baked into client bundles.
2. Fetch and cache it on mount.
3. POST to `{endpoint}/v0.5/query` with `Authorization: Bearer <key>` from the browser directly.

Observed in SWaT demo: server proxy path was hanging on `/query` calls for 90–150s while a direct probe (Node→Newton, same payload) responded in 3.5s. Going direct from the browser restored that 3.5s latency.

See [references/suggested-actions-prompt.md](references/suggested-actions-prompt.md) for a full worked example including the direct-call implementation.

## Debounce & Caching

`/query` calls are not free (latency + billing). If your query input is a function of upstream state that flaps frequently:

- **Hash the input** (e.g., sort the anomalous set into a signature string `"P1,P3,P4"`). Skip the call if it matches the last-queried signature.
- **Debounce** by 1–2s so rapid flapping collapses into a single call. Track in-flight state to avoid restarting the debounce while a previous call is still running.
- **Refetch on signature drift after the in-flight call settles**, not while it's still pending — preserves consistency if the anomaly set moves during a long-running query.

## Testing Prompts In Isolation

Before wiring up the full app, test your prompt with a standalone Node probe:

```js
// probe-query.js
import { readFileSync } from 'fs';

const env = Object.fromEntries(
  readFileSync('.env', 'utf-8')
    .split('\n')
    .map((l) => l.match(/^([A-Z_][A-Z0-9_]*)=(.*)$/))
    .filter(Boolean)
    .map((m) => [m[1], m[2]])
);

const res = await fetch(`${env.ATAI_API_ENDPOINT}/v0.5/query`, {
  method: 'POST',
  headers: {
    Authorization: `Bearer ${env.ATAI_API_KEY}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    query: '...', system_prompt: '...', instruction_prompt: '...',
    file_ids: [], model: 'Newton::...', max_new_tokens: 700, sanitize: false
  })
});
console.log(await res.text());
```

Iterate on prompt text here, not in the browser, until you're getting the shape you want.

## Common Failure Modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Output wrapped in ` ```json ... ``` ` | Newton defaulted to markdown | Fence-stripping in the parser (always do this regardless) |
| Cites a familiar-looking sensor instead of the actual anomaly | Newton followed priors over state | Pre-pick server-side, emphatic "cite this" rule |
| Copies your example values verbatim | Concrete identifiers in the prompt | Use placeholder names (`ZZZ000`, `PX`) in examples |
| Drops half the expected fields | No separator or missing multi-part spec | `part 1 — part 2` format with verb list |
| Routes outputs to wrong target (e.g. upstream → P1 when it should be P2) | Topology implicit, not stated | Inline topology table + server-side validation |
| `JSON.parse` crashes on punctuation | `sanitize: true` rewrote quotes/dashes | Set `sanitize: false` |
| Output truncated mid-sentence | `max_new_tokens` too low | Bump to 1.2× your observed p95 output length |
| First call slow (5–10s), subsequent fast | Model warmup | Accept; pre-warm at startup with a no-op query if cold-start matters |

## Minimal Skeleton

```js
const SYSTEM_PROMPT = `<your rules + shape + constraints + examples with placeholders>`;

function buildQuery(state) {
  // Render `state` into structured text. Include "cite this" hints if pre-picking.
  return `Current state:\n${formatState(state)}\n\n<restate output instruction>`;
}

async function callNewton(state) {
  const res = await fetch(`${endpoint}/v0.5/query`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: buildQuery(state),
      system_prompt: SYSTEM_PROMPT,
      instruction_prompt: SYSTEM_PROMPT,
      file_ids: [],
      model: 'Newton::<your_checkpoint>',
      max_new_tokens: 700,
      sanitize: false
    })
  });
  const data = await res.json();
  const raw = data.response?.response?.[0] ?? '';
  const parsed = parseStructured(raw);
  if (!parsed) throw new Error(`Unparseable: ${raw.slice(0, 200)}`);
  return validateTopology(parsed);  // your server-side validator
}
```

## Building a frontend on top of this

If you're wrapping `/query` output in a React/Svelte/etc. UI — operator suggestion panel, structured-decision dashboard, AI-reasoning sidebar — **read [`DESIGN.md`](../../DESIGN.md) at the root of this repo before writing any CSS**. The Archetype design system (Tailwind v4 + `@archetypeai/ds-lib-tokens` + PP Neue Montreal sans/mono + OKLCH palette + dark-first) is the expected visual language for these demos. The [`SuggestedActions` panel in `newton-swat-demo`](https://github.com/archetypeai/newton-swat-demo) is the canonical reference for surfacing structured `/query` JSON in an operator-facing UI (per-action card with `good`/`warning`/`critical` Badge, mono numeric thresholds, sharp 2px radii); [`newton-wifi-demo`](https://github.com/archetypeai/newton-wifi-demo) shows the per-window verdict + reason pattern. Setting this up at the start is much cheaper than retrofitting later.
