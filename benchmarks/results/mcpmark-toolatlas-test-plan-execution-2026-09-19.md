# MCPMark ToolAtlas test-plan execution — Phase 0 + Phase 1 + Phase 2 report

Date (UTC): 2026-09-19 (updated: Phase 2 integration implemented and tested)
Plan: `benchmarks/MCPMARK_TOOLATLAS_TEST_PLAN.md` (634 lines, prepared 2026-09-19)
Scope of this file: **plan execution through Phase 2: pin/manifest, passing
Phase 1 stock smoke (1 task, k=1), and the tested thin integration. The
canary and the main 10/20 ToolAtlas A/B have not started.**
Result: pre-registered manifest + verified upstream pin + passing stock lifecycle
proof + tested integration + remaining wrapper work.
This is **not** an official MCPMark leaderboard score, not a full 127-task result,
not a paper reproduction, and not a cost-reduction claim.

Claim label (pre-registered, no evidence claimed yet):

> On a pre-registered, held-out subset of MCPMark Verified Filesystem tasks at
> commit `cd45b7f57923b9b3985467f5139927575f83141c`, the same model and MCPMark
> agent were evaluated with ToolAtlas memory disabled versus a frozen ToolAtlas
> memory trained only on disjoint tasks. Outcomes were scored by the official
> task verifiers over four fresh attempts per task.

Do not use this file as: leaderboard score, full-suite result, paper reproduction,
cross-environment result, or proof of lower cost.

## 1. What was done (strictly per plan)

1. Re-resolved the pinned upstream revision live (GitHub API + raw fetch), rather
   than trusting local copies.
2. Verified the plan's inventory: **127 standard tasks total; 30 Filesystem tasks
   across 10 categories.**
3. Verified the concrete correctness fix the plan requires: upstream Verified
   `time_classification` asks for **mtime**, while the local copied task still
   asks for **ctime** and must not be treated as authoritative.
4. Generated and locked the deterministic **10 training / 20 evaluation**
   Filesystem split exactly by the plan rule, plus the paired 40-training /
   160-evaluation attempt schedule — before any model execution.
5. Built a local image from the pinned source and ran the Phase 1 stock smoke
   (`file_property/size_classification`, k=1) in Linux Docker — **PASSED 1/1**
   with the official verifier (see §4A).
6. Audited Phase 2–5 integration gaps against the repo working tree.
7. Re-stated existing custom-harness numbers as **diagnostic only**, per the
   plan's executive decision (they are not the primary ToolAtlas evidence).
8. Re-ran offline validation: `pytest -q` (now 34 passed, 1 GitHub skip,
   including 22 new integration/freeze tests) and
   `python -m toolatlas.paper_benchmark` (metrics identical to the stored
   default; only timestamp/latency fields changed).
9. Implemented and tested the Phase 2 thin integration (§5): agent package,
   `0001-toolatlas-agent.patch` (`git apply --check` clean), read-only memory
   profile, freeze helper, and both test suites (22 ToolMem + 8 in-image).

No custom harness (`mcpmark_file_property_ab`, `mcpmark_postgres_ab`,
`paper_benchmark`, `docker/mcpmark_replay`) was run as new evidence for this
report. Existing saved outputs are cited with their limitations intact. The
`paper_benchmark` rerun only reconfirms the deterministic control artifact.

## 2. Phase 0 — freeze scope and source

### 2.1 Upstream pin

- Repository: `https://github.com/eval-sys/mcpmark`
- Full SHA: `cd45b7f57923b9b3985467f5139927575f83141c`
- Commit date: 2026-06-12 (Verified default; README announces "MCPMark Verified
  is now the default" at this revision)
- Suite: `standard`
- Lock file: `benchmarks/mcpmark/upstream.lock.json`
- Image: local build `mcpmark-pinned:cd45b7f` (image ID
  `sha256:7df46d78ac9f…`; built 2026-09-19 from a clean shallow checkout of the
  pinned SHA). All official Docker Hub tags predate the Verified commit (newest
  push 2025-09-20; image content has no `standard/` suite), so they were
  rejected per the plan's pin-by-digest rule. A floating `latest` tag was not
  used for the reportable run.

### 2.2 Verified inventory (live, at pinned SHA)

| Service | Standard tasks | Categories |
|---|---:|---|
| Filesystem | 30 | 10 |
| GitHub | 23 | 6 |
| Notion | 28 | 10 |
| Playwright | 4 | 2 |
| Playwright-WebArena | 21 | 3 |
| Postgres | 21 | 7 |
| **Total** | **127** | **38** |

Filesystem breakdown (matches plan's 30-task / 10-category statement):

- `desktop` (3): `music_report`, `project_management`, `timeline_extraction`
- `desktop_template` (3): `budget_computation`, `contact_information`, `file_arrangement`
- `file_context` (5): `duplicates_searching`, `file_merging`, `file_splitting`, `pattern_matching`, `uppercase`
- `file_property` (2): `size_classification`, `time_classification`
- `folder_structure` (2): `structure_analysis`, `structure_mirror`
- `legal_document` (3): `dispute_review`, `individual_comments`, `solution_tracing`
- `papers` (3): `author_folders`, `find_math_paper`, `organize_legacy_papers`
- `student_database` (3): `duplicate_name`, `english_talent`, `gradebased_score`
- `threestudio` (3): `code_locating`, `output_analysis`, `requirements_completion`
- `votenet` (3): `dataset_comparison`, `debugging`, `requirements_writing`

Category directory SHAs at the pin (from the contents API; full values retained
for hash validation):

- `desktop`: `03b66fc8f71780646c671abba2f3dd5f27941377`
- `desktop_template`: `ae547ad42f56d277723d699c263bede8418c7dfd`
- `file_context`: `9a41e5aedd6c74c60d1b4e49d4f95e100a928667`
- `file_property`: `56a2e1b0c32ad9dde75bbfbfeaf617c6f20bfe9c`
- `folder_structure`: `29363ee5ef2ff7a3805bb9bcf0d8b26316fb2d1e`
- `legal_document`: `9731b834a4ca6b5114de054bd48a5eea6e0f1cef`
- `papers`: `48aa834208a622338e7f5750320111af9b126b68`
- `student_database`: `1473f64073a4ef83b498ffbd34a419b0c44af399`
- `threestudio`: `80383029886f310b5bb382b71827c530229d3f37`
- `votenet`: `216e9f7d26d40ef55b059d08eee3b9cfe2a2c1de`

### 2.3 Gate: mtime vs stale ctime copy — PASS (upstream verified, local quarantined)

- Upstream raw `file_property/time_classification/description.md` at the pin
  requires **last modified time (mtime)**, explicitly assuming China Standard
  Time (UTC+8), with `metadata_analyse.txt` reporting oldest/latest mtime.
- Local `benchmarks/official/file_property/time_classification/description.md`
  still asks for **creation time (ctime)**. It is stale and must not be used as
  authoritative. No copied local task file was treated as authoritative in the
  split or manifest.
- Local asset fingerprints (for provenance, not as task authority):
  `benchmarks/official/file_property.zip` (~31 MB, SHA-256 prefix `99d5449c…`,
  matching the saved harness snapshot prefix) and
  `benchmarks/official/postgres/chinook.backup` (~163 KB, prefix `50c7969a…`).

## 3. Phase 3 (pre-registration) — locked 10/20 split and paired schedule

Rule applied exactly: `SHA256("toolatlas-mcpmark-filesystem-v1" + task_id)`,
lowest hash per category → training; remainder → evaluation.

Artifacts:

- `benchmarks/mcpmark/manifests/filesystem-verified-v1.json`
- `benchmarks/mcpmark/manifests/filesystem-verified-v1.schedule.json`

Training set — 10 tasks (one per category):

- `desktop/project_management`
- `desktop_template/contact_information`
- `file_context/file_splitting`
- `file_property/size_classification`
- `folder_structure/structure_mirror`
- `legal_document/dispute_review`
- `papers/organize_legacy_papers`
- `student_database/gradebased_score`
- `threestudio/code_locating`
- `votenet/debugging`

Evaluation set — 20 tasks (disjoint IDs, same-service/category-covered):

- `desktop/music_report`, `desktop/timeline_extraction`
- `desktop_template/budget_computation`, `desktop_template/file_arrangement`
- `file_context/duplicates_searching`, `file_context/file_merging`,
  `file_context/pattern_matching`, `file_context/uppercase`
- `file_property/time_classification`
- `folder_structure/structure_analysis`
- `legal_document/individual_comments`, `legal_document/solution_tracing`
- `papers/author_folders`, `papers/find_math_paper`
- `student_database/duplicate_name`, `student_database/english_talent`
- `threestudio/output_analysis`, `threestudio/requirements_completion`
- `votenet/dataset_comparison`, `votenet/requirements_writing`

Schedule (precommitted, no results seen):

- Training: 10 tasks × 4 fixed independent attempts = **40 attempts**.
  Fixed count — never "train until success and stop".
- Evaluation: 20 tasks × 4 attempts × 2 arms = **160 agent attempts**.
- Arm order per `(task_id, attempt)` uses one hash bit of
  `SHA256(prefix + ":order:" + task_id + ":" + attempt)`; paired arms run close
  together with official state reset and a fresh conversation between them.
- Every attempt requires a unique immutable ID and output directory; failures,
  timeouts, verifier failures, and infrastructure errors are preserved in the
  denominator. Resume must refuse any model/provider/task/agent/memory/prompt/
  budget/code/image/schema mismatch.
- Model, provider, reasoning, budgets, prompt/guidance-template hashes, spending
  caps: **intentionally TBD — locking any of them requires a new experiment ID.**
  No model runs were performed under this manifest.

Label: **held-out-task, same-service/category-covered split.** MCPMark categories
are not paper environment instances. An optional held-out-category proxy
(4-category train / 6-category eval) may follow, but must be labeled as a proxy,
not a paper cross-environment reproduction.

## 4. Phase 1 — stock official smoke (PASSED 1/1, 2026-09-19 rerun)

Executed in Linux Docker from the pinned source (Docker Desktop 4.41.2,
linux/amd64; `--memory=4g --cpus=2`):

```bash
python3 -m pipeline --mcp filesystem --task-suite standard \
  --tasks file_property/size_classification \
  --models openai/moonshotai/kimi-k3 --agent mcpmark \
  --exp-name stock-filesystem-smoke-r6 --k 1 --timeout 900
```

Ingested artifacts: `benchmarks/results/mcpmark/stock-filesystem-smoke/`
(`summary.json`, `task-meta.json`, `task-messages.json`, `task-execution.log`,
`provenance.json`) — secret/host-path scanned, clean (container paths only).

| Gate item | Result |
|---|---|
| Official setup (30 Filesystem tasks discovered, disposable task root) | PASS |
| Provider MCP startup (`server-filesystem@2025.12.18`, 14 tools) | PASS |
| Agent execution (`MCPMarkAgent`, 6 turns, tools used, finish `stop`) | PASS |
| Token accounting (22,525 total / 21,141 in / 1,384 out) | PASS |
| Independent official verifier (dirs, classification, empty root, sizes, count 8) | **PASSED** |
| Cleanup (backup dir removed) | PASS |
| Official artifacts present | PASS |
| Credentials/host paths absent from shareable files | PASS |

Headline: **tasks passed 1/1 (100%), total 744.9s (agent 744.1s).**
This is a lifecycle smoke test only, not ToolAtlas evidence.

Two failed pre-runs are preserved in the analysis (fresh experiment IDs, never
resumed across configs): r1/r2 failed at the model layer because the pinned
agent unconditionally sends Moonshot-only body params (`enforcer_mode`,
`think_mode`) and the NVIDIA OpenAI-compatible endpoint rejects them with 400
(`LITELLM_DROP_PARAMS` cannot help — litellm only auto-retries on 422).
Resolution, fully recorded in `provenance.json`: a host-side sanitizing
forward-proxy stripped exactly those two keys per call and injected the real
upstream key server-side, so the eval container held only a dummy key. R3/R4
also exposed two proxy bugs of ours (unwired POST handler; doubled `/v1`
prefix); R6 is the clean run (9 forwarded calls, all HTTP 200; agent made
6 turns). The proxy is harness plumbing outside the repo (temp dir), identical
treatment would apply to both arms of any future A/B, and the model/provider
choice becomes part of the locked manifest when the main experiment starts.

## 5. Phase 2 — thin ToolAtlas integration (IMPLEMENTED 2026-09-19)

Delivered under `benchmarks/mcpmark/` (package, patch, scripts, tests) plus
two ToolMem runtime additions. Full doc: `benchmarks/mcpmark/README.md`.

- `ToolAtlasMCPMarkAgent` registered as `--agent toolatlas`, bound to the real
  `MCPMarkAgent` (no model/provider logic copied). Interception reuses
  official execution: `_create_mcp_server` (trace wrapper),
  `_execute_litellm_with_tools` / `_execute_claude_native_with_tools`
  (guidance/rollout capture, signatures identical).
- Explicit flags: `--toolatlas-mode off|learn|read`, `--toolatlas-memory`,
  `--toolatlas-trace-dir`, `--toolatlas-top-k`, `--toolatlas-read-budget`.
  `off` = stock path, 0 memory calls; `read` = exactly 1 read-only
  `get_guidance` per attempt, fixed block injected only when non-empty;
  `learn` = no calls during execution, post-verifier ingestion only.
- MCP-boundary tracing (`TracedMCPServer`): events recorded before awaiting
  the provider and updated with result/exception — raised calls are counted
  with their error. Discovery (`list_tools`) is separate from invocations.
- Post-verifier hook `on_verified_rollout(...)` runs after the official
  verifier and before cleanup; hook failures return a reason and never break
  the run. Verified successes → affordance evidence; failures → boundary
  evidence; all content sanitized (no secrets/paths/literals/answers).
- Memory is a real stdio subprocess with a sanitized env (DB path + mode flag
  only; credentials never inherited). Evaluation uses the new read-only server
  profile (`get_guidance`, `inspect_tool`, `suggest_probes`, `refresh_status`,
  `memory_stats` only).
- Freeze path implemented (`python -m toolatlas.freeze_memory`): WAL
  checkpoint/truncate → SQLite backup API → `integrity_check` → SHA-256 →
  read-only logical stats.
- Pinned diff: `patches/0001-toolatlas-agent.patch` (9 files, +876/−1,
  `git apply --check` clean). Applied via `scripts/apply_integration.py`
  (exact-anchor edits, fails loudly on drift).
- Validation: ToolMem `pytest` 34 passed / 1 skipped (22 new); in-image wiring
  suite 8 passed (registry, CLI, evaluator forwarding, deterministic
  off-equivalence loop); `toolatlas-demo` clean stats reconfirmed on a fresh
  DB (3/3/1/3/3/0). One design correction is recorded: the first mixin draft
  overrode a 4-arg loop signature that does not exist on the real agent
  (caught by the in-image test); interception now matches official signatures.

Remaining gaps (Phases 3–5): JSON schemas for manifest/attempt records; the
paired-run wrapper (`run_training` / `run_paired_ab` / `aggregate` /
`audit_results`); the non-reportable canary; the full 10/20 A/B (40 training +
160 evaluation attempts with model spend). The pre-registered manifest and
schedule from §3 are unchanged and still locked.

## 5A. Paper mechanisms implemented in the repo (2026-09-19)

To make the memory itself save calls (not just the harness around it), the
paper's §3 mechanisms were implemented deterministically in `src/toolatlas/`:

- Intent induction: structural rationales upgrade to positional intent;
  tips distill from successes (planning) and failures (fix pairs).
- Hybrid retrieval: lexical first, trigram fallback at 0.15 (paraphrases
  rescued; unrelated queries still return empty — verified in tests).
- Normalized, near-sequence-tolerant strategy merge (modulo ≤1 step).
- Guidance carries `avoid` cautions, conventions-first ordering, token
  estimate + optional `token_budget`, coverage, and retrieval provenance;
  playbook capped at 8 steps with early stop of extra tool reads.
- Offline explorer (`explorer.py`): allow-listed, destructive-refused,
  dry-run capable; verified confirmations → affordances, rejections →
  boundaries, contradictions → `unexpected` (never ingested).
- Refresh scheduling via `reverification_due` (also on the read-only server).
- Validation: 21 new mechanism tests green; full suite 55 passed / 1 skipped;
  `toolatlas-demo` clean stats reconfirmed (3/3/1/3/3/0); paper-benchmark
  control regenerated with identical metrics (1.0/1.0, 96→48, 96→72).

## 6. Existing numbers — diagnostic only (not ToolAtlas evidence)

Per the plan's executive decision, the following are engineering checks. They do
not use the official lifecycle end to end and must not be cited as the
ToolAtlas A/B result.

### 6.1 Deterministic local control (`paper_benchmark`, tracked JSON)

- Scope label: `local_filesystem_control_not_full_paper_reproduction`.
- Stored default: 24 runs per arm per split; pass@1/pass@4 = 1.0/1.0.
- Provider calls 96 → 48 (−50%); total MCP calls 96 → 72 (−25%) including one
  `get_guidance` per assisted run. Always report both, never the provider-only
  reduction alone.
- Training is a separate offline cost; break-even after 10 evaluation runs.
- Proves structural call elimination for this deterministic task family only —
  not LLM-agent effectiveness. Baseline includes explicit discovery calls.

### 6.2 Gemini 2.5 Flash file-property slice (16 attempts, same-task reuse)

- Size: 1/4 → 4/4 verified successes. Time: 0/4 → 0/4. Combined 1/8 → 4/8.
  Pass@4 across the two tasks is 50% → 50% (both arms solve size at least once;
  neither solves time).
- Provider calls: 72 → 66 (size), 68 → 80 (time), 140 → 146 (both).
- Recorded total MCP calls: 72 → 70, 68 → 84, 140 → 154 (memory adds 8 calls).
- Recorded tokens (prompt+candidates only, incomplete usage): 67,016 → 58,867;
  98,960 → 91,470; 165,976 → 150,337.
- Elapsed: ~313s → ~348s (size slower despite fewer calls); ~565s → ~196s;
  ~878s → ~544s total. Arms ran sequentially without counterbalancing.
- Size failures show output-placement mismatch (extra `test/` directory; the
  verifier expects category folders directly under the task root). Time failures
  are confounded by the stale local ctime-vs-verifier mismatch described above
  (assisted runs organize by creation-date `09/17`; the verifier expects
  fixed July/August buckets). The saved verifier logs contain machine-local
  paths and need redaction before any publication.
- Accounting limits preserved from the prior analysis: training actually ran
  twice per task (44 size + 58 time = 102 provider calls) while only the second
  run's counts were reported; exception-raising calls may be missing from
  totals; token totals omit thinking/cached/other fields; resume does not
  validate model/provider/settings; no before/after frozen-DB hash or complete
  trajectories were retained.

### 6.3 Postgres Chinook partials

- `postgres-chinook-gemini25flash-k4/attempts_partial.jsonl` holds 3 rows
  (baseline attempt present); `postgres-chinook-gemini35flash-k4` holds 0 rows.
  Incomplete — no conclusion. Same-task-reuse custom harness; diagnostic only.

### 6.4 Offline Docker validation (2026-09-17)

- 44 tools exercised (10 Memory, 3 Text, 14 Filesystem, 17 Everything),
  pytest 12 passed / 1 GitHub skip, 4/4 official size-task deterministic replays
  passed. Useful isolation record for the prototype; not a live official
  MCPMark agent evaluation and not evidence of Docker isolation for host-run
  live A/B results.

Supported diagnostic sentence only: ToolAtlas improved size-task reliability on
one small same-task-reuse sample; it did not reduce total recorded calls across
the whole run, and the time-task result is confounded by timestamp semantics.

## 7. Acceptance checklist (plan § Acceptance checklist)

- [x] Upstream commit pinned (`cd45b7f…`, 2026-06-12).
- [x] 127-task and 30-task Filesystem inventories match the pin (live API).
- [x] Official tasks/verifiers/state assets pass hash validation — category SHAs
  recorded; local image built from the pinned SHA recorded by image ID in
  `upstream.lock.json` (no registry digest exists for a local build).
- [x] Local stale copied tasks quarantined (ctime copy documented, not used).
- [x] Stock official Filesystem smoke succeeds end to end — **PASSED 1/1**
  (r6, 2026-09-19; artifacts ingested under
  `benchmarks/results/mcpmark/stock-filesystem-smoke/`).
- [x] Both arms use the same ToolAtlas-aware MCPMark agent — implemented
  (`ToolAtlasMCPMarkAgent`, registered `toolatlas`; deterministic
  off-equivalence proven in-image; no model runs through it yet).
- [x] Baseline 0 memory calls / assisted exactly 1 retrieval per attempt —
  enforced by mode gating and unit-tested (fake provider/memory suites).
- [x] Memory writes impossible during evaluation — read-only server profile
  (write tools unregistered; error-result + tool-absence tests).
- [x] Freeze path exists (`freeze_memory`: checkpoint → backup API →
  integrity_check → hashes); frozen-hash stability before/after evaluation
  awaits the first real run.
- [x] Training/evaluation IDs disjoint; split + order committed before execution.
- [ ] Clean-state + fresh-conversation isolation — specified, not yet executed.
- [ ] Model cannot access verifier/answers — specified, not yet executed.
- [x] Raised/failed calls recorded at the MCP boundary (tracer); complete
  token/latency/cost retention across training+evaluation awaits the wrapper.
- [ ] Resume rejects provenance mismatches — not implemented (wrapper work).
- [ ] Failures in denominator; reports reproducible from ledger — specified,
  wrapper pending.
- [x] No secrets or absolute host paths in this report (relative paths only).
- [x] Wording stays within claim scope (no leaderboard/full/paper/cost claim).

## 8. Definition of done — not met

The plan is complete only when one fresh, pinned Filesystem experiment produces
a fully auditable matched A/B over the pre-registered 10/20 split (four attempts
per evaluation task, official state + verification, read-only frozen memory,
complete attempt records) with no claim beyond the evidence. That run has not
started. Next gated steps:

1. ~~Start Docker Desktop; build/pin the image~~ — DONE 2026-09-19 (engine up,
   `mcpmark-pinned:cd45b7f` in `upstream.lock.json`; Phase 1 smoke passed).
2. ~~Implement the thin `toolatlas` agent patch~~ — DONE 2026-09-19
   (`benchmarks/mcpmark/`: package + `patches/0001-toolatlas-agent.patch`,
   22 ToolMem unit tests, 8 in-image wiring tests, all passing).
3. ~~Run the stock `k=1` smoke~~ — DONE 2026-09-19 (1/1 PASSED); next run the
   non-reportable canary (3 train / 6 eval / 1 attempt per arm) and discard
   canary artifacts. Canary and full runs need the paired-run wrapper
   (training/eval scheduling, provenance-checked resume, aggregation) plus a
   model-spend decision.
4. Execute training (40 attempts) → freeze/hash → paired evaluation
   (160 attempts) in a fresh experiment directory without code/prompt/memory
   changes; on any defect, preserve the run, bump protocol version, restart.
5. Generate JSON/JSONL/CSV/Markdown from immutable records; run secret and
   host-path scans; report pass@1/pass@4/pass^4/avg@4, paired deltas with
   bootstrap CIs, provider/memory/total calls, discovery counts separately,
   full token/latency/cost fields, training cost + amortization point,
   empty-retrieval rate, and failure categories including infrastructure.

## 9. Audit note

- Scanned this report for credential patterns (`API_KEY`, `TOKEN`, `sk-`,
  `Bearer`) and absolute host paths (drive-letter and `/home/`, `/tmp/`
  patterns): none present. The smoke model alias and proxy plumbing are
  recorded in `benchmarks/results/mcpmark/stock-filesystem-smoke/provenance.json`;
  main-experiment model, budgets, and costs remain TBD until locked.
- Prior saved verifier logs under `benchmarks/results/mcpmark/` contain
  absolute Windows paths — redact before any publication; they were summarized
  here, not quoted.
- Raw databases stay under ignored `.toolatlas/`; only the sanitized manifest,
  schedule, inventory, and hashes above are publishable.

## Source index (verified live for this report)

- Pinned Filesystem standard tree and README at `cd45b7f…` (10 categories,
  Verified-default notice).
- Pinned corrected mtime `time_classification/description.md` (raw fetch).
- Contents API for `tasks/filesystem/standard`, each category, and the
  `tasks/` service roots used for the 127-task count.
- Local files: `benchmarks/official/file_property/time_classification/
  description.md` (stale ctime), `benchmarks/README.md` (dangling smoke
  reference), `src/toolatlas/storage.py` (WAL-only), `src/toolatlas/
  memory_server.py` (no read-only profile), tracked
  `benchmarks/results/paper-protocol-filesystem.json`, saved
  `benchmarks/results/mcpmark/file-property-gemini25flash-k4/report.json` and
  `attempts_partial.jsonl` (16 rows), Postgres partials (3 + 0 rows),
  `benchmarks/results/docker-validation-2026-09-17.md`, and
  `benchmarks/results/live-results-analysis-2026-09-17.md`.
