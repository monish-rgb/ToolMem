# Analysis of saved live results

The newest completed artifact is [Gemini 2.5 Flash file-property A/B](mcpmark/file-property-gemini25flash-k4/report.json): two tasks, four attempts per arm, temperature 0. Its 16 unique attempt records exactly match `attempts_partial.jsonl`. Snapshot SHA-256 matches the harness pin. This analysis reads existing output and code; it does not start paid inference or alter original results.

## Measured results

Numbers below are sums across four attempts per task, baseline → ToolAtlas.

| Metric | Size classification | Time classification | Both tasks |
|---|---:|---:|---:|
| Verified successes | 1/4 → 4/4 | 0/4 → 0/4 | 1/8 → 4/8 |
| Provider calls | 72 → 66 | 68 → 80 | 140 → 146 |
| Memory calls | 0 → 4 | 0 → 4 | 0 → 8 |
| Recorded total MCP tool calls | 72 → 70 | 68 → 84 | 140 → 154 |
| Recorded tokens | 67,016 → 58,867 | 98,960 → 91,470 | 165,976 → 150,337 |
| Elapsed seconds | 312.7 → 347.9 | 565.3 → 195.7 | 878.0 → 543.6 |

Size classification improved observed per-attempt success by 75 percentage points. Provider calls fell 8.3%, total calls fell only 2.8% after guidance overhead, and recorded tokens fell 12.2%. Elapsed time increased 11.3%: fewer calls did not imply a faster run.

Across both tasks, success increased from 12.5% to 50%, but provider calls increased 4.3% and total calls increased 10%. Recorded tokens fell 9.4%. These are descriptive results from two tasks, not evidence of a general call-reduction rate. Calls and tokens include failed attempts, so the arms do not represent equal amounts of successful work. Wall time varies widely and arms run sequentially without counterbalancing.

Under the paper's metrics, observed pass@1 is 12.5% → 50% across these two tasks. Observed pass@4 is **50% → 50%**: both arms solve size at least once, and neither solves time. Four independent fresh-state attempts do not imply a four-task sample or statistically established improvement.

## Why attempts passed or failed

**Size: output placement is the visible difference.** All three failed baseline attempts put category folders under an extra `test/` directory. The official verifier checks `small_files/`, `medium_files/`, and `large_files/` directly under the provided root. Saved directories and final responses corroborate that mismatch. The fourth baseline and all four assisted attempts pass. This supports improved adherence to the intended workspace layout in this sample; it does not establish that the failures came from inability to compare file sizes. Complete argument/response trajectories were not saved, so the exact causal effect of the playbook cannot be reconstructed.

**Time: the environment and task contract disagree.** The official description asks for creation times; task metadata describes modification times. The verifier hardcodes July/August 2025 buckets. Restore preserves ZIP modification times through `os.utime`, but newly extracted files have new creation times. For example, saved `sg.jpg` has a September 17, 2026 creation timestamp and a July 9, 2025 modification timestamp. All assisted attempts organize files into `09/17`, consistent with their reported creation dates, and fail the historical-date verifier. The last baseline does the same; earlier baseline attempts create dummy files instead of organizing the supplied files.

The scripted training policy explicitly selects `modified`, while its stored rationale only says to read metadata for categorization. Thus training can pass without transferring the key timestamp distinction. The 0/4 scores are real verifier failures, but cannot fairly be attributed solely to model or memory quality. The verifier also prints some successful subchecks when expected directories are absent because those subchecks skip missing directories; rely on its final exit code, not individual green messages.

## Validity and accounting limits

- **Training/test overlap:** both exact tasks are trained twice on the same snapshot later used for evaluation. There is no disjoint task split. This is same-task reuse with frozen memory, not the paper's held-out same-environment or cross-environment experiment.
- **Custom harness:** Filesystem is real stdio MCP; memory is an in-process MCP client/server. Official task assets are used, but this is not the official MCPMark agent pipeline. Saved paths and verifier errors show native Windows execution; the previous offline Docker security record does not apply to this live run.
- **Training cost undercount:** `train_task()` returns the last run's call count after performing two runs. Recorded 22 size + 29 time corresponds to 44 + 58 = **102 provider training calls**, before memory ingestion/registration. No amortized cost advantage is established by the report.
- **Evaluation call undercount risk:** `log.calls.append()` happens after the MCP call returns. Calls that raise exceptions are absent from totals, and the saved records do not identify their count. Discovery/handshake messages are also excluded. Call these recorded tool invocations, not every protocol request.
- **Token scope:** native Gemini reporting records `promptTokenCount` and `candidatesTokenCount` only. It does not persist the complete usage object or separate thinking/cached-token fields. The table supports a reduction in these recorded tokens, not a complete billable/inference-token comparison or the paper's RQ4 replication.
- **Resume provenance:** resume loads existing rows by task/arm/attempt without checking model, provider, temperature, code, or memory identity. Rows contain no model field. The report declares Gemini 2.5 Flash, but cannot independently prove every resumed row used that configuration. Fresh output directories are needed when settings change.
- **Frozen memory evidence:** evaluation code contains no ingestion/reverification, but the report does not store a before/after database hash, complete retrieved guidance, or full tool trajectories. These would strengthen reproducibility.

## Other saved runs

The older [Kimi read-only Filesystem result](../../benchmark%20-%201%20result%20read%20only%20FS.txt) reports 3/3 successes in each arm and 9 → 7 provider calls. The harness retrieves guidance once per task: including those three calls gives **9 → 10 total evaluation tool calls**, an 11.1% increase, before training. Its advertised 22% saving concerns provider calls only. Evidence counts of 26 across two source queries suggest accumulated memory evidence; this artifact does not demonstrate a fresh-memory comparison.

`benchmark - 2 github tool.txt` is empty. `nim-complex.json` contains partial stderr/HTTP logs, not a completed JSON result. `fp-k1.log` and `fp-k4.log` contain progress only. `fp-gemini35-k4.log` ends in an HTTP 400 reporting a missing `thought_signature`; that is an API/tool-conversation failure, not a completed benchmark score. The current native Gemini loop preserves raw response parts, but this alone does not demonstrate a successful rerun of that failed configuration. The old Kimi MCPMark smoke JSON/Markdown files are deleted in the current working tree and were not used for this analysis.

## Next experiment

1. Validate timestamp semantics before spending on more time-task attempts. Preserve the official assets and current failures. If a modified-date task is used, label it an adaptation and apply identical instructions to both arms; do not quietly change the verifier to accept the current output.
2. Use disjoint training/test tasks and fresh output/memory directories, recording provider, model, code revision, snapshot/verifier hashes, and frozen-memory hashes.
3. Record all attempted calls, arguments, results/errors, complete guidance, full usage metadata, and both training runs. Redact credentials and machine-local paths before sharing.
4. Rerun matched arms in the intended isolated environment, alternate their order, and report success together with provider calls, memory calls, tokens, and latency. More tasks are needed before claiming generalization.

The supported conclusion is: **ToolAtlas improved size-task reliability on this small same-task-reuse sample. It did not reduce total recorded calls across the whole run, and the time-task result is confounded by timestamp semantics.**
