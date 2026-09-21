# ToolAtlas – Agent Guide

Minimal, local prototype of ToolAtlas provider-side memory (arXiv:2607.11126). Teaching demo, not a full paper reproduction: deterministic induction + lexical similarity instead of LLM proposer/reflection/embeddings.

## Commands (Windows PowerShell – always use `.venv`)

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m pytest tests/test_memory.py -q
.\.venv\Scripts\python -m pytest tests/test_paper_benchmark.py -q
.\.venv\Scripts\python -m compileall -q src tests
.\.venv\Scripts\toolatlas-demo
.\.venv\Scripts\python -m toolatlas.paper_benchmark
powershell -NoProfile -File benchmarks/docker/run.ps1
```

The Docker command creates its own Linux `.venv`; Docker Desktop must be running. It installs dependencies and downloads a checksum-pinned snapshot at build time, then runs offline. Results go to a new `benchmarks/results/toolatlas-benchmark-<timestamp>/` directory.

Recreate env: `python -m venv .venv; .\.venv\Scripts\python -m pip install -e ".[dev,nim]"`. Requires Python >=3.11. Only core runtime dep is `mcp>=2.0,<3` (`pyproject.toml`); the `nim` extra adds `openai>=1.0`. Rerun the editable install after adding or renaming console scripts because editable metadata does not update entry points automatically.

## Architecture

- `src/toolatlas/text_tools_server.py`: ordinary MCP tools `normalize_text`, `word_count`, `keyword_count`. Empty keyword raises `ToolError` — intentional boundary probe, not a bug.
- `src/toolatlas/filesystem_demo.py`: end-to-end integration with the pinned official Filesystem MCP server. It must remain restricted to `mcp-sandbox` (or a test temporary directory).
- `src/toolatlas/everything_demo.py`: end-to-end integration with the pinned official Everything MCP server (protocol test incl. roots/sampling/elicitation). Subprocess gets a sanitized env so `get-env` cannot reveal credentials.
- `src/toolatlas/github_demo.py`: read-only integration with the pinned official GitHub MCP server (`@modelcontextprotocol/server-github@2025.4.8`, 26 tools, 14 registered). PAT scoped to one test repo via `GITHUB_TEST_REPOSITORY`; write-capable tools are never called (`READ_ONLY_GITHUB_TOOLS`).
- `src/toolatlas/llm_ab_nim.py`: true LLM A/B on Filesystem (`--workspace complex_workspace` default; `readonly_workspace` via flag). Neutral baseline prompt; only Arm B gets the playbook. 3 tasks.
- `src/toolatlas/llm_ab_everything.py`: true LLM A/B on Everything (3 echo+sum tasks; 2-call floor means easy tasks tie).
- `src/toolatlas/llm_ab_github.py`: true LLM A/B on GitHub (3 count tasks; expected counts resolved by independent direct read at runtime).
- `src/toolatlas/memory_paths.py`: fresh timestamped database paths under `.toolatlas/` unless `--memory` reuse is intentional.
- `package.json` pins `@modelcontextprotocol/server-filesystem`, `server-everything`, `server-github`; `.mcp.json` and `.vscode/mcp.json` configure Filesystem + ToolAtlas Memory as the repository defaults (GitHub omitted: needs a PAT).
- `src/toolatlas/memory_server.py`: `create_memory_server(path, read_only=False)` factory + module-level `mcp` reading `TOOLATLAS_MEMORY_PATH` (default `.toolatlas/memory.db`) and `TOOLATLAS_READ_ONLY`. Tools cover single/batch ingestion, reflected induction (`induce_reflected`, write-only), guidance (`token_budget`, `embed_mode`), probes, refresh, re-verification scheduling (`reverification_due`), re-verification, governance, inspection, and stats. Read-only profile exposes retrieval/inspection only (`READ_ONLY_TOOLS`); write tools are unregistered there, not merely guarded.
- `src/toolatlas/memory.py`: lifecycle-aware `ToolMemory`; shortest successful rollout is the trace backbone. Structural rationales are upgraded to positional intent at induction; tips distill from successes and failures. `induce()` also accepts LLM-reflected `task_level_tips`/`step_rationales` (re-sanitized, cross-tool references rejected via `known_tools`) labeled `induction="llm-reflected"`. Guidance includes provenance, confidence, expiry filtering, avoid-notes, conventions-first ordering, token estimate/budget, coverage, and an explicit traversal audit. Seed retrieval is lexical with a trigram fallback (`TRIGRAM_FALLBACK_THRESHOLD = 0.15`); `embed_mode="embedding"` with an explicit embedder blends versioned embedding cosine with lexical scores and records the mode in the audit. Strategy merge is normalized and near-sequence tolerant.
- `src/toolatlas/storage.py`: SQLite/WAL persistence. Use one writer service per database in production. `embeddings` sidecar table caches provider vectors keyed by `(qid, model, version)`; version mismatches read back as missing so retrieval falls back to lexical.
- `src/toolatlas/llm_client.py`: explicit LLM seam for construction (`LLMCall` protocol, `GeminiLLMCall`, `OpenAICompatLLMCall`, `FakeLLMCall`). Only `llm_call_from_env()` touches credentials and it only runs when called directly — tests and the offline demo use the fake.
- `src/toolatlas/llm_proposer.py`: LLM seed-task/probe designer with strict-JSON parsing, tool-allowlist grounding, and secret/literal scrubbing (`ProposalError` on violation; builder falls back counted or raises).
- `src/toolatlas/reflection.py`: trace reflection (one rationale per step, tips) and capability aggregation; cross-trace tool references and unsupported inventions (trigram < 0.3 vs. observed candidates) are rejected.
- `src/toolatlas/embeddings.py`: `HashEmbedder` (deterministic, credential-free, frozen-safe) and opt-in `ProviderEmbedder` (Gemini/OpenAI-compatible). Only same-`(model, version)` vectors compare.
- `src/toolatlas/offline_builder.py`: deterministic builder plus feature-flagged `build_llm_memory()` (LLM seeds + reflected induction + optional embedding cache, `OfflineCost` with `llm_*`/`proposal_fallbacks` accounting).
- `src/toolatlas/guidance_render.py`: canonical compact renderer (Plan 1) — sequence + ≤2 avoid notes + verification step, post-render cap with head-step/verify floor, plus active discovery suppression directive. Mirrored self-contained in the MCPMark `sanitize.py` formatter.
- `src/toolatlas/tool_filter.py`: dynamic tool-schema pruning (`filter_tools_by_playbook`, `estimate_schema_tokens`). Filters tool schemas down to playbook tools + always-include safety fallbacks (`read_file`, `list_directory`) + `avoid` notes. Eliminates ~4,200 dead-weight tokens per turn on Filesystem and ~2,500 on PostgreSQL.
- `src/toolatlas/history_compress.py`: smart conversation-history compression (`compress_tool_result`) for directory listings and file read outputs, preserving verbatim errors and truncation markers.
- `src/toolatlas/result_limit.py`: shared deterministic tool-result cap (Plan 2 ablation) — generous limit, verbatim errors, explicit truncation marker, identical in both arms.
- `src/toolatlas/llm_memory_demo.py`: offline paper-pipeline demo (`python -m toolatlas.llm_memory_demo`; `--live` needs explicitly exported credentials).
- `src/toolatlas/similarity.py`: lexical `tokens()` + `cosine_text()` (unchanged semantics) plus `normalized_tokens()` (small synonym map), trigram `trigram_similarity()`, and `hybrid_similarity()` backoff. `src/toolatlas/explorer.py`: offline capability exploration (allow-listed, destructive-refused by default, dry-run planning); only verified outcomes ingest (confirmations → affordances, rejections → boundaries, contradictions → `unexpected`). `src/toolatlas/freeze_memory.py`: WAL checkpoint → SQLite backup API → integrity check → hashes.
- `src/toolatlas/demo.py`: real stdio subprocess flow via `mcp.Client` + `StdioServerParameters`: list → `register_tools` → 2 verified `normalize_text→keyword_count` learns → 1 empty-keyword probe → `get_guidance`. Writes `demo-memory.db`.
- `tests/test_memory.py`: pure `ToolMemory` persistence/traversal. `tests/test_mcp.py`: in-process `Client(server)` (no subprocess, unlike demo).
- `tests/test_real_filesystem_mcp.py`: live stdio test of the locally installed official Filesystem server, including sandbox-denial verification.
- `src/toolatlas/readonly_benchmark.py` and `tests/test_readonly_ab.py`: live read-only deterministic A/B comparison. It must never call tools outside `READ_ONLY_TOOLS`.
- `src/toolatlas/paper_benchmark.py` and `tests/test_paper_benchmark.py`: paper-protocol deterministic Filesystem control. It uses a 1:2 train/test shape (3/6), freezes memory before evaluation, runs four attempts per task, covers same- and cross-environment splits, and reports pass@1/pass@4 plus provider, memory, total-MCP-call, latency, and training-amortization metrics.
- `benchmarks/results/paper-protocol-filesystem.json` is the detailed machine-readable benchmark artifact; the adjacent `.md` is the concise report. Default output is intentionally tracked, while benchmark memory databases remain under ignored `.toolatlas/`.
- `benchmarks/docker/`: isolated local validation runner. `tool_coverage.py` exercises every discovered Memory/Text/Filesystem tool over stdio; the Everything demo covers its discovered tools. `mcpmark_replay.py` uses an official snapshot and verifier with a custom deterministic size-classification policy, not an LLM agent.
- `benchmarks/results/docker-validation-2026-09-17.md` records the offline run: 44 tools exercised (10 Memory, 3 Text, 14 Filesystem, 17 Everything), pytest 12 passed/1 GitHub skip, and 4/4 official size-task deterministic replays passed. Raw output is under `toolatlas-benchmark-20260917-002250/`.
- `src/toolatlas/mcpmark_file_property_ab.py`: custom real-LLM A/B harness for official `size_classification` and `time_classification` tasks. Two deterministic verified training runs per task, fresh evaluation snapshots, four attempts per arm by default, and one guidance call per assisted attempt. Filesystem uses stdio; memory uses in-process `Client(server)`. This is separate from the offline Docker runner.
- `src/toolatlas/mcpmark_postgres_ab.py`: custom real-LLM A/B harness for official MCPMark Chinook database `employee_hierarchy_management` against Dockerized PostgreSQL. Evaluates both native Gemini and OpenAI-compatible providers (`moonshotai/kimi-k3`), tracking schema pruning and turn elimination.
- `src/toolatlas/gemini_rest.py`: native Gemini REST adapter; preserves returned model parts for thought-signature round-tripping. Provider selection uses `LLM_PROVIDER`/`LLM_BASE_URL`; credentials remain environment-only. `--preflight` on the file-property harness includes real API requests and is not an offline check.
- `benchmarks/official/`: downloaded file-property snapshot and task descriptions/verifiers. `benchmarks/results/mcpmark/file-property-gemini25flash-k4/report.json` and `attempts_partial.jsonl` contain 16 completed Gemini 2.5 Flash attempts. See `benchmarks/results/live-results-analysis-2026-09-17.md` for interpretation and harness limitations.
- `benchmarks/results/live-benchmarks-summary-gemini-2026-09-21.md` and `benchmarks/results/cross-model-benchmark-analysis-2026-09-21.md` record live multi-model results: -23.7% tokens on Gemini 2.5 Flash and -58.9% tokens on Moonshot Kimi-k3 on PostgreSQL Chinook (exceeding the paper's ~20% RQ4 target), -94.4% tokens on Filesystem size classification, and 10.3x speedup on GitHub MCP triage.
- `tests/fixtures/complex_workspace`: distractor-rich FS for the NIM A/B (staging/US/archive decoys, deploy-checklist runbook). Baseline prompt is neutral on purpose; the search hint lives only in Arm B memory.
- `tests/test_github_mcp.py`: live GitHub test; skips without server install, `GITHUB_PERSONAL_ACCESS_TOKEN`, or `GITHUB_TEST_REPOSITORY`.
- `tests/test_tool_filter.py` and `tests/test_history_compress.py`: unit tests for dynamic tool schema pruning and tool result compression.

## Rules that are easy to break

- Keep MCP stdio end-to-end; do not collapse demo/tests to direct Python calls.
- Do not broaden the Filesystem MCP allowlist beyond `mcp-sandbox` without explicit user approval.
- `.env` is ignored and is not loaded automatically. Export it explicitly for live tests; never make ordinary `pytest` silently trigger paid NIM calls. LLM construction pieces (`llm_call_from_env`, `provider_embedder_from_env`, `--live` demos) only run on direct invocation — tests use `FakeLLMCall`/`HashEmbedder` and must stay offline.
- Never log, print, or commit credentials, including `GITHUB_PERSONAL_ACCESS_TOKEN`, `NVIDIA_API_KEY`, `GEMINI_API_KEY`, or `LLM_API_KEY`. Supply credentials only to processes that require them; memory entries must never contain them.
- GitHub + Everything NIM harnesses stay read-only (`READ_ONLY_GITHUB_TOOLS`; Everything test tools are safe by design). GitHub `get_file_contents` on a missing path raises `MCPError` instead of returning `is_error` — treat the exception as denial.
- Use a fresh `--memory` database per A/B comparison; reuse accumulates evidence counts by design and weakens the comparison.
- GitHub count tasks resolve expected values by independent direct read at runtime; `expected: null` means the count parse failed and only required-tool use was verified — say so, don't claim a count match.
- Demo stderr `Tool 'keyword_count' failed: ... keyword must not be empty` is expected. A Python traceback is not.
- `demo-memory.db*`, `.toolatlas/`, `paper.*` are gitignored.
- `remember_execution` only with verified `resolved` + agent-neutral rationale (intent, not chain-of-thought). Entries must stay environment-invariant: no secrets, PII, literal user data, paths, or agent syntax — `_generic_rationale()` strips paths/quotes/numbers but is not a complete secret/PII filter.
- Capability dedup (`_merge_entry`) is normalized/fuzzy match (exact, case/whitespace, shared-stem ≥12 chars, or trigram ≥0.75 on boilerplate-stripped cores); each entry must keep `source_queries`. Failed rollouts become `boundaries` cautions, not affordances.
- `guide()` returns empty `seed_candidates`/`playbook` when lexical similarity is 0 and trigram fallback is below threshold — do not invent generic advice. Strategies form when the same normalized multi-tool sequence (modulo at most one step) appears in ≥2 traces.
- Tool schema/version/provider changes invalidate dependent memory. Stale, invalid, quarantined, and verification-expired traces must not be served.
- Default verification age is 30 days. Refresh requires an external rerun followed by `reverify_trace`; never mark memory current without verification.
- `set_trace_status` requires a reason and cannot activate memory; only successful `reverify_trace` can reactivate it.
- Clean demo stats: `tools: 3, traces: 3, strategies: 1, trace_edges: 3, executions: 3, stale_traces: 0`.
- Always use a fresh memory database for each benchmark split. Memory must be frozen after training; evaluation runs must never call ingestion or re-verification tools.
- Benchmark call accounting must separate provider calls, memory calls, and total MCP calls. Do not advertise the 50% provider-call reduction without also reporting the 25% total-MCP-call reduction that includes one `get_guidance` call per assisted run.
- The stored default control result is 24 runs per arm per split: pass@1/pass@4 are 1.0/1.0, provider calls are 96→48, total MCP calls are 96→72, and training breaks even after 10 evaluation runs. Regenerate both JSON and Markdown together after benchmark logic or fixture changes.
- Do not overwrite the tracked paper-protocol result with `--runs` other than the default 4. Reduced-run executions are suitable only for tests or local smoke checks.
- The paper benchmark is deliberately labeled `local_filesystem_control_not_full_paper_reproduction`. It proves structural call elimination for this deterministic task family, not production readiness or LLM-agent effectiveness. The baseline includes explicit discovery calls; state that when interpreting the reduction.
- Keep three experiments distinct: the offline deterministic control, the offline official-task replay, and the real-LLM file-property A/B. Historical references to `filesystem-smoke-nvidia-kimi-k3.*` may point to removed files; check artifact existence before citing them.
- The file-property A/B trains on the same two tasks and snapshot used for evaluation. Fresh copies and frozen memory do not make tasks held out. Label it same-task reuse, not the paper's disjoint train/test evaluation. The paper partitions tasks about 1:2 for same-environment evaluation and environment instances about 2:3 for cross-environment evaluation; MCP-Universe is excluded from the latter.
- `time_classification` has a task/environment mismatch: `description.md` requests creation times, `meta.json` describes modification times, and the verifier expects fixed July/August 2025 buckets. Snapshot restore preserves mtimes but creates new birth times; deterministic training uses modified dates. Keep original failures and disclose this mismatch before interpreting scores or running an explicitly labeled correction.
- Audit file-property accounting before cost claims: `train_task()` reports only the second of its two runs (actual provider training calls in the saved run: 44 size + 58 time = 102); tool calls raising exceptions are omitted from `log.calls`; native Gemini token totals currently sum prompt and candidate tokens without recording other usage fields. `total_mcp_calls` counts tool invocations, not discovery/handshake messages.
- `--resume` does not validate model/provider/settings against prior attempts; JSONL rows lack that provenance. Use a fresh output directory when changing any configuration and retain the run's exact configuration. Do not silently pool attempts from different models.
- Preserve Docker isolation: non-root UID, no runtime network or injected credentials, read-only root filesystem, dropped capabilities, no-new-privileges, bounded CPU/RAM/PIDs, no host bind mounts or Docker socket. `.dockerignore` allowlists the build context. The runner copies results out and removes its container/work volume. Host-run live A/B results are not evidence of Docker isolation, and Docker cannot isolate remote account effects if credentials/network are added.
- The paper's RQ4 result measures total inference tokens (3.55M for ToolAtlas versus 4.44M for Vanilla), not raw tool calls. A production claim requires official MCPMark/MCP-Universe tasks and snapshots, four independent LLM rollouts, programmatic verifiers, token accounting, and broader services/agents.
- Ensure deterministic training rollouts use batch primitives (e.g., `list_directory_with_sizes` over iterative `get_file_info`) to prevent inducing $N+1$ iterative playbooks that lock the agent out of batch shortcuts during schema pruning.
- Schema pruning is a double-edged sword when memory encodes an iterative path: strict pruning saves tokens per turn but blinds the agent to un-pruned batch shortcuts. Always ensure tool-family retention or two-tier exposure in production deployments.

## Findings & Lessons: Schema Pruning, Suboptimal Lock-in & Generalization

### 1. The Anomaly: Moonshot Filesystem `size_classification` (22 vs 14 calls)
During the Moonshot Kimi-k3 Filesystem evaluation (`benchmarks/results/eval-fs-moonshotai/kimi-k3`):
- **Baseline**: 14 calls, 21,256 tokens (100% pass).
- **ToolAtlas**: 22 calls, 21,400 tokens (100% pass).

**Root Cause:**
- Deterministic training in `mcpmark_file_property_ab.py` (`_size_policy`) inspected file sizes iteratively using `get_file_info` inside a `for name in files:` loop rather than the single-call `list_directory_with_sizes`.
- ToolAtlas induced a 4-step playbook: `[list_directory -> create_directory -> get_file_info -> move_file]`.
- Dynamic schema pruning (`filter_tools_by_playbook`) pruned away all 9 non-playbook tools, including `list_directory_with_sizes`.
- **Baseline** had all 14 tools in its prompt, discovered `list_directory_with_sizes`, and fetched all sizes in 1 call ($1 + 1 + 3 + 9 = 14$ calls).
- **ToolAtlas** strictly followed its memory playbook, executing 9 separate `get_file_info` calls ($1 + 9 + 3 + 9 = 22$ calls).
- **The Token Paradox**: Despite +57% more calls (22 vs 14), total tokens were flat (+0.6%, 21,400 vs 21,256) because schema pruning cut per-turn prompt overhead from 2,101 to 658 tokens. ToolAtlas actually used *fewer* prompt tokens than Baseline (19,044 vs 19,587); the difference was purely completion tokens for tool arguments.

### 2. Transferability Across MCP Servers (The Single-Item vs. Batch Dilemma)
In empirical benchmarks, this inversion occurred **only** on Filesystem `size_classification`. On PostgreSQL Chinook, Moonshot achieved **-58.9% tokens** and -50% calls; on GitHub triage, Gemini achieved a **10.3x speedup**; and on Filesystem `time_classification`, ToolAtlas saved **-25.9% tokens**.

However, the underlying dynamic ("Single-Item vs. Batch Tool") exists across all real-world MCP APIs:
- **Notion MCP**: `append_block_children` (1 call for $N$ blocks) vs. looping `append_block` ($N$ calls).
- **GitHub MCP**: `list_issues` with filters (1 call) vs. search + looping `get_issue` ($N+1$ calls).
- **PostgreSQL MCP**: A single `JOIN` / `IN (...)` query vs. an iterative $N+1$ `SELECT` loop.
- **Slack MCP**: Bulk digest posting vs. individual message spamming.

If memory records an iterative loop and schema pruning eliminates the batch tool, the agent is locked into the slower path.

### 3. Architectural Safeguards
To prevent lock-in while preserving token reduction across all MCP servers:
1. **Tool-Family Expansion (`tool_filter.py`)**: Expand playbook pruning to include functional sibling tools (e.g., metadata tools `get_file_info` + `list_directory_with_sizes`, or query tools `execute_query` + `describe_table`).
2. **Two-Tier Schema Exposure**: Inject full JSON schemas (~400 tokens each) only for playbook tools, while providing a compact 1-line text catalog (~50 tokens total) of remaining server tools so the agent retains discovery awareness.
3. **Dynamic Fallback Escalation**: If an agent hits a tool error or stalls, dynamically restore the full tool catalog to enable self-healing and alternate paths.
4. **Shortest-Trace Competition (`memory.py`)**: When multiple rollouts succeed, ToolAtlas automatically preserves the shortest trace backbone, allowing faster paths to displace slower historical rollouts.
5. **Training Policy Optimization**: Ensure synthetic training harnesses always exercise batch primitives where available.

## Verify

Rerun `pytest -q` after touching `memory.py`, `models.py`, either server, `similarity.py`, fixtures, or benchmark code. Run `toolatlas-demo` after changing the MCP loop or persistence. Run `python -m toolatlas.paper_benchmark` after changing benchmark logic or its fixtures, then inspect both stored result files and confirm they contain no secrets or machine-local paths.

Current full suite: 132 passed, 1 skipped (GitHub live test skips without server install, PAT, or test repo), including `tests/test_llm_memory.py` (23 LLM-pipeline tests, all faked/offline), `tests/test_tool_filter.py` (14 schema pruning tests), `tests/test_history_compress.py` (5 history compression tests), and `tests/test_notion_server.py` (4 Notion simulator tests). `npm install` is required to execute rather than skip the real Filesystem/Everything integrations.

After changing the Docker harness, run `benchmarks/docker/run.ps1` and inspect `checks.json`, pytest skips, tool coverage, verifier output, and both control report files. Preserve failed attempts and distinguish skipped/unexecuted experiments from task failures. Scan shareable results for secrets and host paths; current live A/B verifier logs contain absolute Windows paths and need redaction before publishing. Python dependency ranges are not locked; the saved Docker run includes `python-packages.txt` and `image-id.txt` captured separately.
