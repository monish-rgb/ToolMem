# Process for benchmarking ToolAtlas beyond Filesystem

Prepared 2026-09-17. This is a setup and implementation runbook; the steps below have not been executed for the seven additional services.

## 1. Scope and current support

Test the same model and agent **without memory (baseline)** and **with frozen ToolAtlas memory (assisted)** against official tasks and verifiers. Allow task-required writes only inside disposable benchmark resources.

| Benchmark | Service | What it tests | Repository support today |
|---|---|---|---|
| MCPMark | GitHub | Repository, issue, branch, PR, and workflow operations | Read-only demo and small count-task A/B; official task adapter still needed |
| MCPMark | Notion | Pages, databases, querying, editing, organization | Adapter and isolated workspace setup needed |
| MCPMark | PostgreSQL | Schema discovery, SQL, data changes | Adapter and official database restores needed |
| MCPMark | Playwright/WebArena | Browser actions on shopping, administration, and forum sites | Adapter and website snapshots needed |
| MCP-Universe | Location Navigation | Places, directions, distances, route constraints | Google Maps service and adapter needed |
| MCP-Universe | Financial Analysis | Financial data retrieval and calculations | YFinance service and adapter needed |
| MCP-Universe | Web Search | Search, page retrieval, multi-source answers | Search/Fetch services and adapter needed |

These are the seven remaining domains selected by the [ToolAtlas paper, Appendix A](https://arxiv.org/html/2607.11126v1#A1). Everything is useful protocol coverage, but is not a paper benchmark domain. Do not substitute application connectors or invented tasks for the official tasks when claiming benchmark results.

The existing `benchmarks/docker/run.ps1` remains the offline Filesystem/Memory/Text/Everything validation runner. It does **not** launch the seven services above. The existing file-property LLM harness is a useful implementation reference, but its same-task training, incomplete accounting, and resume provenance need correction before reuse. See [the live-results analysis](results/live-results-analysis-2026-09-17.md).

## 2. Prepare one reproducible experiment

1. Choose a model/provider already available to you, a spending limit, agent framework, temperature, output-token limit, maximum turns, and per-attempt timeout. Keep these identical across arms. A model different from the paper is a prototype evaluation, not an exact reproduction.
2. Check out MCPMark and MCP-Universe into separate directories. Pin full commit IDs; pin MCP server versions, container digests, dependency locks, task files, and snapshot hashes. Keep each framework's dependencies in its own `.venv` or image to avoid MCP SDK conflicts.
3. Use the setup documentation from those pinned revisions. Current upstream instructions may differ from the older digest-pinned MCPMark image already in this repository.
4. Inventory task IDs, their environment-instance IDs, required tools, credentials, expected output, and official verifier. Resolve concrete IDs from the checkout; placeholders in this document are not task names.
5. Define training and test manifests **before running either arm**. Save the split seed and task IDs. Match the paper's released manifests if available; otherwise label your split independently reconstructed.
6. Create a fresh experiment directory and memory database. Never resume into another model/configuration's output directory.

Suggested output layout (new structure to implement):

```text
benchmarks/results/<service>/<experiment-id>/
  manifest.json             # versions, model settings, split, hashes; no secrets
  train.jsonl               # every training attempt and verifier outcome
  attempts.jsonl            # every evaluation attempt, including errors
  guidance/                 # actual task-conditioned guidance sent to Arm B
  traces/                   # sanitized calls, arguments, results, model usage
  report.json
  report.md
.toolatlas/<experiment-id>/ # private memory DBs; do not publish raw databases
```

## 3. Docker design for live services

Create a **separate live-service Compose configuration**; do not weaken the existing offline runner. This configuration and the adapters are implementation work, not files already supplied by this runbook.

| Component | Access needed | Isolation boundary |
|---|---|---|
| LLM agent | Selected model endpoint; MCP connections | No host filesystem, Docker socket, database admin credentials, or unrelated API keys |
| ToolAtlas memory server | Training DB during training; frozen evidence during evaluation | No cloud service credentials; separate DB per experiment |
| Service MCP server | Only its benchmark service | Only its dedicated credential and resource scope |
| Reset/verifier controller | Official snapshots and benchmark state | Separate from agent; model cannot read verifier code or reference answers |
| PostgreSQL/WebArena | Private Docker network and disposable volumes | No public ports; snapshot reset before each attempt |
| Egress gateway | Explicit external API destinations | Network policy enforced outside the agent container |

Use non-root execution, dropped capabilities, `no-new-privileges`, bounded CPU/RAM/PIDs, and a read-only runner filesystem with designated writable work/output locations. Size browser/database resources separately; do not assume the offline runner's limits fit them. Use Docker's default seccomp profile; do not solve browser errors by enabling privileged mode or disabling every sandbox.

For database/browser services, use an internal Docker network. Let only a dedicated gateway reach the internet for model calls. GitHub, Notion, Maps, YFinance, and web retrieval also need controlled outbound access. A normal Docker bridge or `HTTP_PROXY` variable alone does not enforce a destination allowlist. Implement firewall/proxy enforcement and verify that a non-permitted destination fails. Search/Fetch needs a broader policy than Maps: deny loopback, private networks, cloud metadata endpoints, and redirects to them while allowing task-required public pages.

Keep credentials out of Dockerfiles, build arguments, image layers, prompts, memory, logs, and tracked files. Inject secrets only into processes needing them. The provider MCP subprocess must not inherit the model API key through a broad environment copy. Docker isolation does not prevent remote GitHub/Notion writes: dedicated test accounts and resources provide that boundary.

## 4. Service-by-service setup

The numbered actions below describe the proposed workflow. Links identify upstream setup requirements; verify names against your pinned revision.

### A. GitHub

1. Create a dedicated evaluation organization and import the official exported repository states. Set `GITHUB_EVAL_ORG` and `GITHUB_TOKENS` for the upstream runner. Scope access to benchmark resources and the operations required by selected tasks. [MCPMark GitHub setup](https://github.com/eval-sys/mcpmark/blob/main/docs/mcp/github.md)
2. Choose one official task. Provision a disposable repository instance, execute the baseline, and verify state using the official verifier.
3. Restore a clean instance for every training/evaluation attempt. Confirm issues, PRs, branches, and commits from the previous attempt cannot leak into the next one.
4. Add the common ToolAtlas adapter described in section 5; expand to held-out tasks only after baseline setup and cleanup work.

**Existing local smoke path:** with a scoped PAT and `GITHUB_TEST_REPOSITORY` already exported, these commands run from ToolMem on the host. They are read-only checks, **not Docker-isolated official MCPMark runs**:

```powershell
.\.venv\Scripts\python -m pytest tests/test_github_mcp.py -q
.\.venv\Scripts\python -m toolatlas.github_demo
```

Keep `READ_ONLY_GITHUB_TOOLS` intact. Implement official write-task support separately with disposable repositories. A null expected count in the existing count harness does not establish answer correctness.

### B. Notion

1. Prepare separate source and evaluation hubs from the official templates, with separate integrations. Configure `SOURCE_NOTION_API_KEY`, `EVAL_NOTION_API_KEY`, `SOURCE_PARENT_PAGE_TITLE`, and `EVAL_PARENT_PAGE_TITLE`. Follow upstream browser authentication setup; its UI automation expects English. [MCPMark Notion setup](https://github.com/eval-sys/mcpmark/blob/main/docs/mcp/notion.md)
2. Give the agent access only to the disposable evaluation copy. Keep source-hub credentials in the reset controller.
3. Validate duplication, integration access, and the official verifier on one task before evaluation.
4. Restore from source templates each attempt, allowing for API propagation before verification. Store resource IDs privately for cleanup, never in reusable memory.

### C. PostgreSQL

1. Run a dedicated database container and restore the official database backups. Current upstream docs use pgvector/PostgreSQL 17 and provide employee, Chinook, DVD Rental, sports, and Lego backups. Match client/server versions to the pinned benchmark. Configure its `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USERNAME`, and `POSTGRES_PASSWORD`. [MCPMark PostgreSQL setup](https://github.com/eval-sys/mcpmark/blob/main/docs/mcp/postgres.md)
2. Address the database by Compose service name, not host `localhost`. Do not publish its port publicly.
3. Start with one official task whose required extensions and snapshot are present. Run the official verifier against an independent connection.
4. Restore the whole required state before every attempt; a transaction rollback alone may miss committed changes, sequences, roles, or schema changes. Use an evaluator/admin role for restore and a separate task-appropriate role for the agent.

This is a useful first new service: it supports controlled snapshots and deterministic state verification without a remote workspace account.

### D. Playwright / WebArena

1. Provision the benchmark's shopping, shopping-admin, and Reddit images as required by your selected tasks. Configure their base URLs so the browser and verifier see the same sites. MCPMark distinguishes `playwright` and `playwright_webarena`; choose the service used by the pinned task. [MCPMark Playwright setup](https://github.com/eval-sys/mcpmark/blob/main/docs/mcp/playwright.md)
2. Keep websites on the private benchmark network. Use a fresh browser profile/context for each attempt; never mount your personal browser profile.
3. Restore website data as well as cookies/session state. Restarting only the browser does not reset orders, posts, or admin edits.
4. Run a login/navigation smoke check, then one official task and its state verifier. Screenshots and the model saying “done” do not replace verification.

### E. MCP-Universe: Location Navigation

1. Configure `GOOGLE_MAPS_API_KEY` and the chosen model credential, with quotas set for the experiment. Use the domain's `location_navigation.yaml` configuration and official task evaluators. [MCP-Universe setup](https://github.com/SalesforceAIResearch/MCP-Universe#evaluating-llms-and-agents)
2. Check one geocoding/directions task end-to-end, including units, coordinates, transport mode, and requested time semantics.
3. Use disjoint train/test tasks. Start a fresh conversation and clear task-specific state per attempt; the live Maps service itself cannot be reset like a database.
4. Run matched arms close together, log observation timestamps and responses, and preserve evaluator configuration. Distinguish transport/API failures from task-answer failures.

### F. MCP-Universe: Financial Analysis

1. Provision the YFinance MCP service used by the selected `financial_analysis.yaml` tasks. Give it network access to the required data endpoints, separate from the model credential. Inspect the pinned service for any additional configuration requirements.
2. Confirm symbols, currency, interval, timezone, adjustment settings, and market-data dates against the task evaluator before paid runs.
3. Execute matched arms in the same observation window and record source timestamps. Do not let Arm B see more recent data without disclosing it.
4. Use the official evaluator. If live-data drift makes a task unevaluable, report that condition. A recorded-data replay can be useful, but label it an adaptation rather than a live benchmark.

### G. MCP-Universe: Web Search

1. Provision the configured Google Search and Fetch servers for `web_search.yaml`. Upstream lists `SERP_API_KEY`; its environment template also supports other search routes. Match the exact provider in the pinned task/server configuration instead of changing search engines silently. [Environment template](https://github.com/SalesforceAIResearch/MCP-Universe/blob/main/.env.example)
2. Check one task requiring both search and fetch, and validate its official output evaluator.
3. Preserve returned URLs, timestamps, content hashes, fetch errors, and final structured answer. Enforce the public-web boundary described in section 3.
4. Reset conversations/caches per attempt or apply an identical documented cache policy. Do not store answers, personal data, or fetched page text in reusable ToolAtlas memory.

## 5. Add ToolAtlas to the official runners

This adapter does not yet exist for all seven services. Do not assume a `--toolatlas` flag exists upstream.

Implement these operations for each domain: `prepare_instance`, `reset_instance`, `connect_mcp`, `execute_agent`, `verify`, and `cleanup`. These are proposed adapter interfaces, not existing import names.

Training:

1. Connect to the real service MCP endpoint and register discovered schemas, provider identity, and version with ToolAtlas.
2. Execute training tasks only. Save every attempted call, including errors and retries, and use the independent official verifier to mark outcomes.
3. Ingest verified rollouts with agent-neutral, environment-invariant rationales. Failures become boundary cautions. Strip literal answers, secrets, IDs, account names, and workspace-specific locations.
4. Account for every training rollout, probe, model token, provider call, and memory call. A deterministic teacher policy is a prototype variant; label it rather than calling it the paper's LLM-based memory construction.
5. Create a consistent frozen snapshot using SQLite backup/checkpoint semantics; do not copy only an active WAL database's main file. Preserve a canonical hash and prohibit evidence mutation during evaluation. If the current server needs writable SQLite files on startup, use isolated copies and verify their logical memory contents remain unchanged.

Evaluation pseudocode:

```text
for each held-out task:
  for attempt in 1..4:
    alternate or randomize which arm goes first
    for arm in [baseline, assisted] in that order:
      reset to the task's official initial state
      create a fresh agent conversation
      use identical model, provider tools, task, and budgets
      if assisted:
        query frozen ToolAtlas with task text (never reference answers)
        record guidance call and the exact injected guidance
      execute agent through MCP
      verify independently with the official evaluator
      persist outcome, attempted calls, full usage, latency, and errors
      clean up only this attempt's resources
```

If lexical retrieval returns no relevant guidance, leave it empty. Do not insert hand-written task-specific hints to rescue Arm B. Use identical prompt scaffolding and the same injection policy throughout the experiment; the intended difference is memory guidance.

The adapter should invoke ToolAtlas over stdio as in the repository demos, and keep the provider's official transport. Never replace the provider with direct SDK/database calls inside the model's task loop. Reset and independent verification may use direct administrative APIs.

## 6. Commands and where they run

### Existing offline validation — ToolMem, Windows PowerShell

```powershell
powershell -NoProfile -File benchmarks/docker/run.ps1
```

This makes no live cloud API benchmark calls.

### Official MCPMark baseline — prepared Linux runner container

Inside the pinned MCPMark checkout, with its `.venv` and benchmark resources configured:

```bash
.venv/bin/python -m pipeline --help
```

Command templates below require substitution of `MODEL_ALIAS` and `CATEGORY/TASK` with actual configured values. They are baseline invocations, not ToolAtlas-assisted commands:

```bash
.venv/bin/python -m pipeline --mcp github --models MODEL_ALIAS --tasks CATEGORY/TASK --exp-name github-smoke --k 1
.venv/bin/python -m pipeline --mcp notion --models MODEL_ALIAS --tasks CATEGORY/TASK --exp-name notion-smoke --k 1
.venv/bin/python -m pipeline --mcp postgres --models MODEL_ALIAS --tasks CATEGORY/TASK --exp-name postgres-smoke --k 1
.venv/bin/python -m pipeline --mcp playwright_webarena --models MODEL_ALIAS --tasks CATEGORY/TASK --exp-name webarena-smoke --k 1
```

The [upstream pipeline](https://github.com/eval-sys/mcpmark/blob/main/pipeline.py) defines the CLI. Confirm it at the pinned revision. Register your chosen provider/model using that checkout's model configuration; ToolMem's `LLM_MODEL` alone does not configure the upstream runner. After smoke verification, use `--k 4` on your held-out manifest and the implemented A/B adapter.

### MCP-Universe baseline — prepared Linux runner container

From its pinned checkout, configure the corresponding files under `mcpuniverse/benchmark/configs/test/`, including task selection and model provider. The documented domain entry points are:

```bash
export PYTHONPATH=.
.venv/bin/python tests/benchmark/mcpuniverse/test_benchmark_location_navigation.py
.venv/bin/python tests/benchmark/mcpuniverse/test_benchmark_financial_analysis.py
.venv/bin/python tests/benchmark/mcpuniverse/test_benchmark_web_search.py
```

These scripts can select multiple tasks; narrow the configuration to one smoke task before executing. They do not automatically implement ToolAtlas arms, a disjoint training split, or four fresh attempts. Add that orchestration in the adapter. Verify script locations in your pinned revision. [Official execution instructions](https://github.com/SalesforceAIResearch/MCP-Universe#execution)

## 7. Split sizes and rollout stages

The supplied paper's Appendix A gives these same-environment counts for the remaining domains:

| Domain | Train tasks | Test tasks | Test attempts across both arms at k=4 |
|---|---:|---:|---:|
| GitHub | 8 | 15 | 120 |
| Notion | 10 | 18 | 144 |
| PostgreSQL | 8 | 13 | 104 |
| Playwright | 6 | 13 | 104 |
| Location Navigation | 9 | 26 | 208 |
| Financial Analysis | 10 | 30 | 240 |
| Web Search | 13 | 37 | 296 |
| **Total** | **64** | **152** | **1,216** |

Counts are from the supplied paper, not a promise that current upstream tasks are identical. The per-service ratios are approximate and vary. Four training rollouts for these 64 tasks would add 256 attempts before any separate bootstrapping or exploration costs. [Paper tables 5–6](https://arxiv.org/html/2607.11126v1#A1)

Use three stages:

1. **Smoke:** one official task per service, one baseline attempt. Establish connectivity, reset, and verifier correctness; do not claim performance.
2. **Pilot:** disjoint small train/test sets and four attempts per arm. A 3-train/6-test pilot gives 48 evaluation attempts per service; label this your pilot split.
3. **Expanded evaluation:** use all selected paper-aligned task manifests. Then evaluate cross-environment transfer for MCPMark by splitting entire environment instances, roughly 2:3. The paper excludes MCP-Universe from cross-environment evaluation. Do not reuse the same trained database across these independent splits.

For full paper scope, cross-agent transfer, multiple model backbones, competing baselines, exploration, and ablations are further experiments; seven single-model A/B runs alone do not reproduce them.

## 8. Required reporting and completion criteria

For each task/arm report four verifier outcomes, all attempted provider calls, guidance/memory calls, their sum, model request count, complete raw usage metadata, elapsed time, and failure category. Keep MCP initialization/discovery traffic in a separate counter. Include input/output/thinking/cached tokens where available; do not double-count cached subsets or imply missing fields are zero. Monetary cost needs the applicable provider pricing and billing semantics; token count alone is not a dollar estimate.

Compute:

- **pass@1:** mean success across the four attempts, then average across tasks.
- **pass@4:** fraction of tasks with at least one successful attempt.
- **Call reduction:** `(baseline - assisted) / baseline`, reported separately for provider and total tool calls.
- **Training amortization:** all offline construction cost divided by measured per-evaluation savings, in the same units. If savings are zero or negative, no break-even is established.

Separate completed failures, infrastructure/API errors, skips, and unexecuted tasks. Define denominator/retry policy in advance and retain original outcomes. Report both task-level and domain-level results; do not hide a domain's failures inside an aggregate. Add uncertainty intervals when there are enough independent tasks; repeated runs of two tasks do not establish broad generalization.

A domain is complete when both arms have four fresh-state attempts for every selected test task, the official verifier ran on each, memory stayed frozen, logs contain no secrets, and the report includes training cost and all failures. Save exact code/configuration hashes and reject incompatible resumes.

Recommended implementation order: **PostgreSQL → GitHub → Notion → Playwright → Maps → YFinance → Web Search**. This order builds reset/verifier infrastructure before adding live-data variability. Do not schedule the full 1,216-attempt evaluation until the pilot has passed environment checks and produced a credible cost estimate.
