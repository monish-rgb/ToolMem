# ToolAtlas Docker validation — 2026-09-17

Scope: current local prototype, real MCP stdio integrations, deterministic
control, and one official MCPMark task replay. **Not a full paper reproduction.**

## Results

Final raw output: [isolated run](toolatlas-benchmark-20260917-002250/checks.json).

| Check | Result |
|---|---|
| Repository pytest suite | 12 passed; GitHub integration skipped without an injected PAT |
| ToolAtlas memory tools | All 10 discovered tools exercised over stdio |
| Text tools | All 3 discovered tools exercised; output checked |
| Filesystem tools | All 14 discovered tools exercised, including write/edit/move/media; outside-root access denied |
| Everything tools | All 17 discovered tools passed; server environment sanitized |
| Demos and compilation | Passed |
| Same-environment deterministic control | Both arms pass@1=1.0, pass@4=1.0; 24 runs per arm |
| Cross-environment deterministic control | Both arms pass@1=1.0, pass@4=1.0; 24 runs per arm |
| Official MCPMark size-classification task replay | Official snapshot and unchanged verifier; 4/4 fresh-snapshot deterministic attempts passed |

In **each control split**, baseline→assisted provider calls are **96→48 (50%
reduction)**, memory calls are **0→24**, and total MCP calls are **96→72 (25%
reduction)**. Training breaks even at 10 evaluation runs in this task family.
The baseline explicitly performs discovery calls; this measures structural
call elimination, not independent LLM reasoning improvements. See the
[JSON](toolatlas-benchmark-20260917-002250/paper-protocol-filesystem.json) and
[Markdown](toolatlas-benchmark-20260917-002250/paper-protocol-filesystem.md).

The official task replay is a custom deterministic policy, baseline only, with
zero inference tokens. It reads file metadata and moves files through actual
MCP calls. Snapshot creation and independent verification use direct file I/O.
It is not the official LLM pipeline, not a ToolAtlas A/B result, and not a model
pass@4 score. Its raw output contains tool-call counts, elapsed time, verifier
messages, and snapshot/verifier hashes.

## Paper coverage and outstanding work

Read the supplied paper HTML, especially §4 and Appendix A. The paper uses
MCPMark's five services and three selected MCP-Universe domains:

| Paper benchmark/service | What this run establishes | Still required for the paper experiment |
|---|---|---|
| MCPMark Filesystem | Local ToolAtlas control, full tool smoke coverage, official size task replay | Matched LLM baseline/assisted arms over official task splits |
| MCPMark GitHub | Not run in this credential-free container | Disposable benchmark repositories, scoped credentials, runner integration |
| MCPMark Notion | Not run | Dedicated benchmark workspace, API credentials, snapshots |
| MCPMark Playwright | Not run | Paper's WebArena shopping/admin/reddit instances and agent runner |
| MCPMark PostgreSQL | Not run | Restored official database snapshots, isolated database network, agent runner |
| MCP-Universe Location Navigation | Not run | Google Maps credentials and configured official tasks/runner |
| MCP-Universe Financial Analysis | Not run | YFinance service/network access and configured official tasks/runner |
| MCP-Universe Web Search | Not run | Google Search/Fetch services, required credentials, official tasks/runner |

These are unexecuted experiments, not failed tasks. The runtime intentionally
received no service credentials or internet access. API spending clarification
was requested; no paid inference was started. Available application connectors
are not substitutes for the paper's service snapshots and programmatic verifiers.

Protocol differences: the paper uses roughly 1:2 disjoint **task** splits for
same-environment evaluation, but roughly 2:3 **environment-instance** splits
for cross-environment evaluation. MCP-Universe is excluded from the latter.
Each official task requires four independent LLM rollouts from its starting
snapshot. The repository's fixed 3/6-task two-fixture control is smaller.
Cross-agent transfer (SwitchAct→ReAct/CodeAct), ablations, competing baselines,
and inference-token cost comparisons remain untested. The paper's RQ4 numbers
(3.55M vs 4.44M) concern inference tokens, not tool calls.

## Isolation and reproducibility

Run [run.ps1](../docker/run.ps1). The [Dockerfile](../docker/Dockerfile) pins the
MCPMark base image by digest; npm packages follow package-lock.json, and the
official snapshot is checksum-pinned. Python dependencies follow pyproject
ranges (not a fully locked environment). Runtime uses a non-root UID, no
network, no host bind mounts/socket/credentials, read-only root filesystem,
dropped capabilities, no-new-privileges, CPU/memory/PID limits, and a disposable
work volume. The volume and container are removed after output is copied.
The saved container-security.json records actual Docker settings.

Docker constrains local effects; it would not prevent changes to remote
accounts if credentials and network access were later supplied.

Sources: [ToolAtlas paper](https://arxiv.org/html/2607.11126v1),
[official MCPMark](https://github.com/eval-sys/mcpmark),
[official MCP-Universe](https://github.com/SalesforceAIResearch/MCP-Universe).
