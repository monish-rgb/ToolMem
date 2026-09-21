# ToolAtlas Input-Token Reduction Plan

Date: 2026-09-20  
Repository: ToolMem  
Primary objective: demonstrate that frozen provider-side ToolAtlas memory reduces
online input tokens while preserving verifier-confirmed task success.

## 1. Outcome and claim boundary

The initial claim will be deliberately narrow:

> On a pre-registered held-out subset of official MCPMark Filesystem tasks, the
> same model and agent use fewer online input tokens with frozen ToolAtlas
> guidance than without it, while maintaining non-inferior official-verifier
> success.

The first experiment will not claim that ToolAtlas reduces tokens for every MCP
server, model, or agent. Broader claims require replication on GitHub,
PostgreSQL, Playwright, and additional agent frameworks.

The primary endpoint is provider-reported online input tokens. The experiment
will also report total inference tokens, success, provider calls, memory calls,
latency, and offline construction cost. Tool-call reduction is a secondary
outcome, not a substitute for direct token measurement.

## 2. Why ToolAtlas can reduce input tokens

In a conventional MCP rollout, each unnecessary model turn can resend or retain
the system prompt, tool schemas, previous tool results, and accumulated
conversation. Repeated discovery and failed attempts therefore increase input
tokens much faster than the size of a single tool response suggests.

ToolAtlas should reduce this cost by giving the agent a small, task-conditioned
summary before the rollout:

1. which tools are relevant;
2. which order has been verified;
3. which batching or high-information operation to prefer;
4. which known failures or invalid assumptions to avoid; and
5. how to verify the result.

The guidance is beneficial only when the avoided context exceeds the guidance
added to the prompt. The implementation must therefore optimize net online
tokens, not merely make the guidance informative.

## 3. Metric contract

### 3.1 Token fields per model request

Every request must retain the provider's raw usage object and normalize, where
available:

- `input_tokens`
- `cached_input_tokens`
- `uncached_input_tokens`
- `output_tokens`
- `reasoning_tokens`
- `total_tokens`
- `usage_source` (`provider`, `tokenizer_estimate`, or `missing`)
- `provider_request_id`

Never silently convert a missing usage field to zero. If the provider does not
return authoritative input usage, mark the request incomplete and exclude the
run from the primary analysis, while retaining it as a diagnostic artifact.

Provider definitions differ. Record raw usage first, then document the exact
normalization formula for the selected provider. Do not add reasoning tokens to
`output_tokens` when the provider already includes them there.

### 3.2 Per-attempt metrics

For each immutable `(experiment_id, task_id, arm, attempt)` record:

- verifier outcome;
- number of model requests/turns;
- online input, cached input, output, reasoning, and total tokens;
- guidance text and guidance token count;
- provider-tool calls, failed provider calls, and retries;
- memory calls;
- total agent-issued MCP calls;
- elapsed time;
- termination reason;
- model/provider/settings provenance; and
- pre/post environment and frozen-memory hashes.

The primary measurement is:

```text
input_token_saving = baseline_online_input_tokens
                   - toolatlas_online_input_tokens
```

Report both cached-inclusive and uncached/billable input tokens when the provider
exposes both. The primary variant must be fixed in the experiment manifest
before the first evaluation attempt.

### 3.3 Cost scopes

Keep three scopes separate:

1. **Online inference:** evaluation-agent model usage, including the injected
   guidance and all retries.
2. **Offline construction:** task proposal, training rollouts, induction,
   reflection, exploration, and memory compression.
3. **Amortized deployment:** offline construction plus online use across a stated
   number of downstream tasks or agents.

If `M` is offline construction input tokens, and the mean online saving is
`S > 0`, report `ceil(M / S)` as the approximate input-token break-even task
count. Also provide a sensitivity range using the confidence interval for `S`.

## 4. Experimental invariants

The following must be identical between baseline and ToolAtlas arms:

- model and returned model ID;
- provider and API adapter;
- temperature, reasoning effort, and token/turn/time budgets;
- agent implementation and base prompts;
- exposed provider tools and schemas;
- starting task snapshot;
- verifier;
- retry policy;
- context compaction policy; and
- network/container resources.

The only intended arm difference is one compact ToolAtlas guidance block before
the assisted rollout. Tool filtering, schema compression, result truncation, or
prompt caching may later be evaluated as separate factorial ablations, but must
not be silently enabled only in the ToolAtlas arm.

## 5. Phase 0 - lock the evidence contract

### Work

1. Create a versioned experiment manifest derived from
   `benchmarks/mcpmark/manifests/filesystem-verified-v1.json`.
2. Replace the current TBD provider section with exact model, provider, budget,
   prompt, code, patch, image, task, verifier, and snapshot hashes.
3. Pre-register the training/evaluation split, four attempts per arm, paired
   order, exclusion rules, success margin, and statistical analysis.
4. Give every attempt a unique ID before execution.
5. Define whether prompt-cached tokens are included in the primary endpoint.

### Exit gate

- The manifest is valid JSON and immutable once live evaluation starts.
- All task, verifier, snapshot, code, prompt, and image hashes resolve.
- No model attempt has been run under the final experiment ID.

## 6. Phase 1 - complete token telemetry

### Files to change

- `src/toolatlas/gemini_rest.py`
- `src/toolatlas/mcpmark_file_property_ab.py`
- `src/toolatlas/mcpmark_postgres_ab.py`
- `benchmarks/mcpmark/integration/toolatlas_mcpmark/agent.py`
- a new shared token-usage model, preferably `src/toolatlas/token_usage.py`

### Work

1. Introduce one normalized `TokenUsage` structure shared by all adapters.
2. Preserve every provider usage field and the original raw response usage.
3. Count usage per request rather than only as an accumulated run total.
4. Record failed and retried requests; mark whether authoritative usage exists.
5. Add model-request IDs, task/arm/attempt IDs, and timestamps.
6. Produce a prompt anatomy summary:
   - fixed system and agent instructions;
   - provider tool schemas;
   - ToolAtlas guidance;
   - conversation and tool-result history;
   - current task text.
7. Cross-check provider totals against a tokenizer estimate for anomaly detection,
   but use authoritative provider input usage for the main result.

### Tests

- exact normalization for each supported provider usage shape;
- cached-token handling;
- reasoning-token handling without double counting;
- missing usage is never treated as zero;
- retries create separate request rows;
- attempt totals equal the sum of request rows; and
- JSONL resume rejects mismatched model/provider/settings.

### Exit gate

- A baseline canary produces a complete request-level token ledger.
- Aggregate input and total tokens recompute exactly from that ledger.
- No attempt included in the primary analysis has missing authoritative input
  usage.

## 7. Phase 2 - minimize online guidance cost

### Work

1. Pass an explicit `token_budget` through the memory MCP `get_guidance` tool.
2. Start with `top_k=3`, `read_budget=8`, and a hard guidance budget of 384
   estimated tokens.
3. Serialize guidance as compact sections:
   - `preferred_sequence`
   - `tool_tips`
   - `avoid`
   - `verification`
4. Remove duplicate statements, provenance prose, confidence explanations, and
   traversal details from the injected block. Preserve those fields in the
   audit artifact, not in the model prompt.
5. Return empty guidance when retrieval confidence/coverage is below the locked
   threshold. Do not inject generic advice.
6. Make one guidance call per assisted attempt. A second retrieval is forbidden
   in the primary experiment.
7. Measure the actual provider-token increase caused by injection, not only the
   local `guidance_tokens_estimate`.

### Budget tuning study

Before the final experiment, use training tasks only to compare fixed budgets
such as 128, 256, 384, and 512 tokens. Select one budget using a pre-declared
utility rule, then lock it. Do not tune on evaluation tasks.

A suitable training-only utility is:

```text
verified_success_rate - lambda * normalized_online_input_tokens
```

The manifest must lock `lambda` before budget selection.

### Exit gate

- Guidance never exceeds its configured budget.
- Empty/irrelevant retrievals do not add a guidance block.
- Injected guidance contains no credentials, host paths, literal user data,
  chain-of-thought, or environment-specific identifiers.

## 8. Phase 3 - paper-style offline memory construction

The current deterministic induction remains useful as a control, but the
paper-aligned path needs a distinct, feature-flagged builder.

### Bootstrapping defaults

- generate three seed tasks per target tool;
- run four independent rollouts per seed task;
- verify every rollout programmatically;
- preserve the shortest verified successful sequence as the trace backbone;
- use failed rollouts only for sanitized boundary evidence; and
- create agent-neutral positional rationales rather than retaining raw
  reasoning.

### Memory induction

Construct or enrich:

1. **Tool-trace knowledge:** task summary, verified tool/rationale sequence,
   task-level tips, provenance, and confidence.
2. **Tool-capability knowledge:** affordances, boundaries, and recurrent
   co-usage.
3. **Tool-strategy knowledge:** recurring multi-tool plans supported by at least
   two independent verified traces.

LLM-generated entries must be schema-constrained and grounded in execution
records. Unsupported statements are rejected. Exact evidence IDs must remain in
the audit record even though they are omitted from online guidance.

### Capability exploration

Run three exploration rounds. In each round, propose affordance and boundary
tasks for under-covered capabilities, execute them in a resettable sandbox, and
ingest only verified outcomes. Track proposal novelty so later rounds do not
repeat already-covered behavior.

### Exit gate

- Every served statement is traceable to verified evidence.
- Rejected and failed probes cannot become affordances.
- Memory contains no task-state literals, credentials, PII, or machine-local
  paths.
- Offline construction tokens and calls are fully accounted for.

## 9. Phase 4 - retrieval and dynamic traversal

### Work

1. Retain the current lexical/trigram retriever as a deterministic fallback.
2. Add versioned embedding similarity for task and trace summaries.
3. Seed retrieval with the top three trace nodes.
4. Traverse trace, tool, and strategy relationships under a maximum read budget
   of eight operations.
5. Rank knowledge by relevance, verification freshness, confidence, and expected
   token utility.
6. Prefer knowledge that can eliminate a model turn or a large/repeated tool
   result.
7. Compress selected facts deterministically where possible. If an LLM performs
   online navigation or compression, its tokens belong to online ToolAtlas
   cost.
8. Log the full traversal audit separately from the injected guidance.

### Retrieval metrics

- empty-guidance rate;
- irrelevant-guidance rate from blinded review;
- retrieval coverage;
- guidance input-token size;
- provider turns avoided;
- repeated tool calls avoided; and
- negative-transfer rate.

### Exit gate

- Retrieval is deterministic for a frozen memory/configuration unless the
  manifest explicitly selects an LLM navigator.
- Schema, provider, or version changes invalidate dependent entries.
- No stale, quarantined, expired, or invalid entry is served.

## 10. Phase 5 - physically freeze evaluation memory

### Work

1. Thread `read_only` through `ToolMemory` and `SQLiteStore`.
2. Open the evaluation database using SQLite URI `mode=ro` and enable
   `PRAGMA query_only=ON`.
3. Skip directory creation, schema initialization, metadata writes, and WAL mode
   changes in read-only mode.
4. Checkpoint and truncate WAL before freezing.
5. Use the SQLite backup API and run `integrity_check`.
6. Hash the main database before and after evaluation.
7. Reject evaluation if the hash changes or WAL/SHM sidecars appear.
8. Keep all write-capable memory tools unregistered in read-only mode.

### Exit gate

- An operating-system read-only database supports guidance retrieval.
- Mutation attempts fail.
- Pre/post hashes are identical.

## 11. Phase 6 - MCPMark paired runner and artifact auditor

### New commands/modules

Implement equivalents of:

- `run_training`
- `freeze_memory`
- `run_paired_ab`
- `aggregate_results`
- `audit_results`

The runner must pass explicit arm and attempt identifiers into
`maybe_begin_attempt`; relying on the current empty-arm/attempt-zero defaults is
not allowed.

### Runner responsibilities

- reset the official task state before every attempt;
- counterbalance baseline/ToolAtlas execution order;
- create a fresh model/agent session for every attempt;
- perform exactly one guidance call in the assisted arm and zero in baseline;
- invoke the official verifier;
- retain failed and timed-out attempts;
- capture request-level token usage and MCP events;
- prevent output/trace collisions;
- validate configuration on resume; and
- stop when the spending or token cap is reached.

### Auditor responsibilities

- parse every tracked JSON/JSONL artifact;
- confirm the scheduled attempt set is complete;
- verify unique attempt IDs;
- recompute all aggregate numbers from request/event ledgers;
- confirm no evaluation-time memory mutation;
- detect missing usage or verifier results;
- scan shareable artifacts for credentials and absolute host paths; and
- distinguish task failure, model failure, verifier failure, timeout, and
  infrastructure failure.

## 12. Phase 7 - experiments

### 12.1 Offline regression control

Run the existing deterministic paper benchmark to ensure later changes do not
break structural call reduction:

```powershell
.\.venv\Scripts\python -m toolatlas.paper_benchmark
```

This is a mechanism test only and cannot establish real-model token savings.

### 12.2 Three-task live canary

Use three training-only or canary tasks, four paired attempts each. Confirm:

- complete authoritative token usage;
- official verifier operation;
- one/zero memory-call invariant;
- prompt and tool parity;
- state restoration;
- frozen DB immutability;
- no trace collisions;
- no missing failed calls; and
- successful aggregate recomputation.

Do not proceed if any canary invariant fails.

### 12.3 Main Filesystem experiment

Use the pre-registered disjoint split:

- 10 training tasks;
- 20 evaluation tasks;
- four training rollouts per task;
- four baseline and four ToolAtlas attempts per evaluation task; and
- paired, counterbalanced arm order.

This produces 40 training attempts and 160 evaluation attempts. Use a fresh
output directory and memory database. Never resume with changed settings.

### 12.4 Replication ladder

After Filesystem succeeds:

1. GitHub with a dedicated evaluation organization/repository and a fine-grained
   token restricted to it;
2. PostgreSQL with a verified database restore before each attempt;
3. another model family;
4. another agent framework; and
5. disjoint environment-instance transfer.

Each replication gets its own locked manifest and must report its result even if
ToolAtlas increases tokens.

## 13. Statistical analysis and acceptance gates

Use attempts as observations but cluster uncertainty by task. For every paired
task/attempt block calculate:

```text
saving = baseline_input_tokens - toolatlas_input_tokens
```

Report:

- mean, median, p90, and total input tokens per arm;
- absolute and percentage savings;
- task-clustered bootstrap 95% confidence interval;
- distribution of per-task savings;
- pass@1 and pass@4;
- success-rate difference and confidence interval;
- input and total tokens per verified success;
- model turns and provider/memory/total MCP calls;
- offline construction cost; and
- amortized break-even tasks.

### Primary success rule

The experiment supports the narrow reduction claim only when both are true:

1. the lower bound of the task-clustered 95% confidence interval for mean input
   token saving is greater than zero; and
2. ToolAtlas satisfies the pre-registered success non-inferiority margin or
   improves verifier success.

Recommended initial non-inferiority margin: no more than five percentage points
lower success for the 20-task study. A tighter two-point margin requires a much
larger evaluation set and should be used for the expanded confirmation study.

### Secondary success rule

Report whether total inference tokens also satisfy the same positive-saving
criterion. Never substitute estimated guidance size for provider-reported input
usage.

## 14. Failure and rollback criteria

Stop the main experiment and create a new experiment version when:

- the model/provider/settings differ from the locked manifest;
- provider usage fields are incomplete or change semantics;
- an evaluation snapshot fails to restore;
- memory changes during evaluation;
- the agent or verifier code changes;
- attempt identities collide;
- baseline and ToolAtlas expose different provider tools;
- a credential or host path appears in a shareable artifact; or
- a retry policy is applied asymmetrically.

ToolAtlas should be disabled for a production query when retrieval is empty,
below the confidence threshold, stale, or predicted to add more tokens than it
saves. The production integration must always retain a clean no-guidance path.

## 15. Optimization sequence after the first valid result

Optimize one variable at a time on training or development tasks:

1. guidance token budget;
2. top-k traces;
3. traversal read budget;
4. inclusion of boundaries;
5. inclusion of strategy nodes;
6. lexical versus embedding retrieval;
7. deterministic versus LLM compression; and
8. retrieval confidence threshold.

For each change, measure success and provider-reported online input tokens. Do
not select configurations using the locked evaluation tasks.

Tool routing, schema compression, prompt caching, and tool-result summarization
may produce additional savings, but should be separate ablations so the measured
ToolAtlas effect remains identifiable.

## 16. Required deliverables

1. Locked experiment manifest and schedule.
2. Request-level token ledger schema and tests.
3. Complete MCP event ledger.
4. Paper-style offline memory builder with construction-cost accounting.
5. Budgeted guidance serializer.
6. Physically read-only frozen-memory implementation and hash audit.
7. Paired runner, aggregator, and artifact auditor.
8. Canary report.
9. Main machine-readable report and concise Markdown interpretation.
10. Redacted reproducibility bundle containing hashes, versions, prompts,
    schedules, and failure classifications.

## 17. Verification commands

After implementation changes:

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m compileall -q src tests
.\.venv\Scripts\toolatlas-demo
.\.venv\Scripts\python -m toolatlas.paper_benchmark
```

After Docker or official MCPMark harness changes:

```powershell
powershell -NoProfile -File benchmarks/docker/run.ps1
```

Live API canaries must use a fresh output directory and an explicitly exported
environment-only credential. Ordinary `pytest` must never trigger paid calls.

## 18. Final evidence statement template

If all gates pass, use:

> On the pre-registered held-out MCPMark Filesystem evaluation at commit
> `<commit>`, frozen ToolAtlas guidance reduced mean provider-reported online
> input tokens from `<baseline>` to `<toolatlas>` (`<percent>%`, task-clustered
> 95% CI `<low>` to `<high>`), while pass@1 changed from `<baseline>` to
> `<toolatlas>` and remained within the pre-registered non-inferiority margin.
> Memory construction cost `<offline_tokens>` input tokens and the estimated
> break-even point was `<tasks>` evaluated tasks. Results apply to the tested
> model, agent, MCP server, task distribution, and configuration.

If either primary gate fails, report the observed numbers and state that the
experiment did not demonstrate input-token reduction. Do not replace the failed
claim with provider-call reduction or a result from the deterministic control.
