# Mini ToolAtlas

A small, runnable implementation of the core ideas in **ToolAtlas: Learning Once, Reusing Everywhere with Tool-Side Memory** (arXiv:2607.11126).

This is intentionally a teaching prototype, not a reproduction of the paper's full benchmark. It keeps the important architecture:

- **Tool-Trace graph:** verified executions become agent-neutral `(tool, rationale)` traces.
- **Tool-Capability graph:** tools accumulate evidence-backed affordances, boundaries, and co-usage patterns.
- **Tool-Strategy graph:** repeated multi-tool sequences become reusable planning strategies.
- **Dynamic traversal:** a new task seeds retrieval from similar traces and follows graph links under a read budget.
- **Provider-side persistence:** memory is stored in a SQLite/WAL database owned by the MCP tool provider and reused by different clients or agents.
- **Lifecycle management:** schema fingerprints, verification age, confidence, provenance, refresh candidates, and governance status prevent outdated memory from being served silently.

The expensive LLM proposer/reflection and embedding model from the research system are replaced with deterministic rules and local bag-of-words similarity. This keeps the end-to-end mechanism inspectable. The deterministic core is API-key free; only the optional NIM LLM comparisons and the GitHub integration need keys (`NVIDIA_API_KEY`, `GITHUB_PERSONAL_ACCESS_TOKEN`), always via environment variables, never committed.

## Run on Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\toolatlas-demo
```

For the optional NIM true-model comparisons (same model/prompt/temperature both arms):

```powershell
.\.venv\Scripts\python -m pip install -e ".[nim]"
```

The small demo starts two local **stdio MCP servers**:

1. `toolatlas-text-server` provides `normalize_text`, `word_count`, and `keyword_count`.
2. `toolatlas-memory-server` provides `register_tools`, `remember_execution`, `remember_rollouts`, `suggest_probes`, `get_guidance`, `inspect_tool`, `refresh_status`, `reverify_trace`, `set_trace_status`, and `memory_stats`.

It executes two successful composed tasks plus one failing boundary probe, verifies their outcomes, persists `demo-memory.db`, and retrieves guidance for an unseen task.

For a real third-party server integration, install the pinned official Filesystem MCP server and run:

```powershell
npm install
.\.venv\Scripts\python -m pytest tests/test_real_filesystem_mcp.py -v
.\.venv\Scripts\python -m toolatlas.filesystem_demo
```

The real integration scopes Filesystem MCP to `mcp-sandbox`, discovers its live schemas, verifies write/read/list operations, confirms that access outside the sandbox is denied, learns provider-side memory, and retrieves a reusable filesystem playbook.

The server was checked before connection through PolicyLayer. Its identity was verified, but it received grade D because its 14-tool surface includes four write-capable tools and had recently changed. For that reason, this repository pins the tested package version, invokes the installed entry point without runtime downloads, and restricts it to `mcp-sandbox`. See the [registry record](https://policylayer.com/tools/filesystem).

## Everything MCP protocol integration

The pinned Everything server exercises MCP tools plus roots, sampling, elicitation,
resources, logging, and tasks. It is a protocol test server rather than a filesystem
server: `--root` is advertised through MCP roots and used as the subprocess working
directory, but the server does not read or edit that directory.

```powershell
npm install
.\.venv\Scripts\python -m toolatlas.everything_demo --root "C:\path\to\project"
.\.venv\Scripts\python -m pytest tests/test_everything_mcp.py -v
```

The integration calls every tool discovered in the live handshake. Its subprocess
receives a sanitized environment so the server's `get-env` test tool cannot reveal
credentials. Sampling is deterministic, and elicitation is declined automatically;
no browser is opened and no user information is collected.

PolicyLayer reported an unverified identity, grade D, and a recent grade change
for this server; the connection was made only after explicit approval. The
package is pinned and the integration avoids inherited credentials. See the
[registry record](https://policylayer.com/tools/server-everything).

## GitHub MCP integration (read-only)

The pinned `@modelcontextprotocol/server-github@2025.4.8` server is exercised
read-only against a single test repository. Write-capable tools
(create/update/merge/push/...) are never called; a client-side allowlist
(`READ_ONLY_GITHUB_TOOLS`) blocks them. The PAT needs only read scopes on the
test repo and is passed solely to the server subprocess environment — it is
never logged or stored in memory.

```powershell
npm install
$env:GITHUB_PERSONAL_ACCESS_TOKEN="<pat>"
$env:GITHUB_TEST_REPOSITORY="owner/repo"
.\.venv\Scripts\python -m pytest tests/test_github_mcp.py -v
.\.venv\Scripts\python -m toolatlas.github_demo
```

The demo verifies repository access via a direct read (search is not used as a
gate because the search index misses private/forked repos), runs two composed
`list_issues → list_commits` overviews so a strategy forms, probes a missing
file path as a boundary (this server raises `MCPError: Not Found` instead of
returning an error flag, which the harness treats as denial), and retrieves a
`list_issues → list_commits` playbook. Without a token the test skips.

## Live read-only A/B comparison

Run the same verified task with a baseline agent and a ToolAtlas-assisted agent:

```powershell
npm install
.\.venv\Scripts\python -m toolatlas.readonly_benchmark
```

Or run it as a test:

```powershell
.\.venv\Scripts\python -m pytest tests/test_readonly_ab.py -v
```

Both benchmark CLIs create a uniquely named database under `.toolatlas/` when
`--memory` is omitted. Pass `--memory <path>` only when intentionally testing
reuse across runs. The selected database path is included in the JSON output.

The benchmark launches the real pinned Filesystem MCP subprocess against static files in `tests/fixtures/readonly_workspace`. A client-side allowlist permits only read operations and raises immediately on any write-capable tool call.

The controlled task asks both agents to find a deployment policy and report a setting. The current expected comparison is:

| Variant | Result | Filesystem calls | Tool sequence |
|---|---:|---:|---|
| Baseline without ToolAtlas | Pass | 4 | `list_directory → directory_tree → search_files → read_text_file` |
| Agent with ToolAtlas | Pass | 2 | `search_files → read_text_file` |

Training calls used to bootstrap provider memory are reported separately and excluded from the comparison. Wall-clock latency is recorded but not asserted because process scheduling varies.

This is a deterministic control experiment over a live MCP server, not an LLM-quality benchmark. It isolates whether retrieved provider memory can reduce exploration calls. For an LLM comparison, use the same model, prompt, temperature, task set, and verifier in both arms; enable only the ToolAtlas memory server in the assisted arm.

## True LLM A/B comparisons (NVIDIA NIM)

Three harnesses run Arm A (provider tools only) vs Arm B (`get_guidance` once,
then the same tools) with the same model, neutral system prompt, temperature,
tasks, and verifier. The baseline prompt carries no strategy hint; only Arm B
receives the learned playbook. Set credentials per window (PowerShell shown;
in `cmd` use `set VAR=value` instead of `$env:VAR="value"`):

```powershell
$env:NVIDIA_API_KEY="<nim-key>"
$env:NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1"
$env:NVIDIA_MODEL="moonshotai/kimi-k3"
```

Filesystem with distractors (`tests/fixtures/complex_workspace`: staging/US/
archive decoys plus a deploy-checklist runbook; 3 tasks):

```powershell
.\.venv\Scripts\python -m toolatlas.llm_ab_nim --workspace complex_workspace --memory .toolatlas\nim-complex.db --temperature 0
```

Everything protocol server (3 `echo label + sum` tasks, optimal floor is 2
calls, so expect ties on easy tasks):

```powershell
.\.venv\Scripts\python -m toolatlas.llm_ab_everything --root mcp-sandbox --memory .toolatlas\everything-nim.db --temperature 0
```

GitHub read-only (3 count tasks on the test repo; expected counts are resolved
by an independent direct read at runtime because repos change; a `null`
expected value means the count check was skipped and only required-tool use
was verified):

```powershell
$env:GITHUB_PERSONAL_ACCESS_TOKEN="<pat>"
$env:GITHUB_TEST_REPOSITORY="owner/repo"
.\.venv\Scripts\python -m toolatlas.llm_ab_github --memory .toolatlas\github-nim.db --temperature 0
```

Use a fresh `--memory` database per comparison run; reuse accumulates evidence
counts across runs by design. Redirect output to a file (`> run.json 2>&1`)
since `cmd` truncates long output.

## Paper-protocol benchmark

Run the stored, deterministic Filesystem control using the evaluation shape from
the ToolAtlas paper (disjoint 1:2 train/test tasks, frozen memory, same- and
cross-environment splits, four runs per task, pass@1/pass@4, and cost metrics):

```powershell
.\.venv\Scripts\python -m toolatlas.paper_benchmark
```

Results are written to `benchmarks/results/paper-protocol-filesystem.json` and
`.md`. The report separates provider calls from the `get_guidance` memory call,
so it cannot hide retrieval overhead. It also reports memory-construction cost
and the number of evaluation runs required to amortize that cost.

This is a paper-aligned local control, not the full paper reproduction. The full
evaluation requires MCPMark and MCP-Universe environments across eight services,
their task snapshots and verifiers, four independent LLM rollouts per task, and
inference-token accounting.

## Connect the servers to an MCP host

The repository now includes two ready project configurations:

- `.mcp.json` for clients that support the common project MCP format and launch from the repository root.
- `.vscode/mcp.json` for VS Code with `${workspaceFolder}` paths.

Both configurations expose the installed official Filesystem server and ToolAtlas memory server. The toy text server is retained only for the small demo and unit tests. GitHub is intentionally omitted from the committed configs because it requires a PAT; pass the token via the environment when running the GitHub demo/harness.

Equivalent configuration:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "node",
      "args": [
        "node_modules/@modelcontextprotocol/server-filesystem/dist/index.js",
        "mcp-sandbox"
      ]
    },
    "toolatlas-memory": {
      "command": "C:\\path\\to\\ToolMem\\.venv\\Scripts\\python.exe",
      "args": ["-m", "toolatlas.memory_server"],
      "env": {"TOOLATLAS_MEMORY_PATH": ".toolatlas/filesystem-memory.db"}
    }
  }
}
```

Recommended agent flow:

1. Call `get_guidance` before solving a task and place the returned playbook/tips in the agent context.
2. Use the ordinary provider tools.
3. Verify the result externally.
4. Call `remember_execution` with only agent-neutral rationales and the verified outcome.

For repeated attempts of one task, prefer `remember_rollouts`. It chooses a successful backbone, retains corrections from failures, assigns execution IDs, and computes confidence from the verified batch.

## Lifecycle and governance

Each tool receives a SHA-256 fingerprint over its name, description, input schema, provider, and version. Calling `register_tools` with a changed definition marks dependent traces and capability entries stale. Stale, invalid, quarantined, or verification-expired traces are excluded from `get_guidance`.

Use the lifecycle tools as follows:

```text
register_tools(updated schemas)
        ↓
refresh_status(max_age_days=30)
        ↓
rerun each returned task against the current provider
        ↓
reverify_trace(task_id, resolved, verifier_type)
```

`set_trace_status` is the manual governance hook. It requires an audit reason and can set `stale`, `invalid`, or `quarantined`. Only successful `reverify_trace` calls can return knowledge to `active`.

Guidance now includes:

- A versioned response schema.
- Confidence and source provenance.
- Evidence counts for tool tips.
- An explicit `ReadTrace` / `Expand` / `ReadTool` / `ReadStrategy` / `Done` traversal audit.
- Reads used versus the configured read budget.

SQLite is configured in WAL mode and mutations are guarded within a server process. A production deployment should still use one memory-writer service instead of starting several independent writers against the same database.

## Where this differs from the full paper

The paper uses three seed tasks per tool, four rollouts per task, three exploration rounds with three boundary and three affordance probes per target, semantic embeddings, and an LLM navigator with an eight-read budget. This prototype supports the same data flow and the default `top_k=3`, `read_budget=8`, but the included demo is deliberately smaller. See `src/toolatlas/memory.py` for the induction/traversal logic and `src/toolatlas/demo.py` for the execution-verified MCP loop.

## Remaining improvements

The lifecycle work addresses changing APIs, storage, re-verification, and basic governance, but these production concerns remain:

1. Replace lexical similarity with a pluggable embedding backend while keeping the offline fallback.
2. Run generated verifiers inside an isolated sandbox with time, network, and filesystem limits.
3. Add tenant/provider namespaces, authentication, and authorization around governance tools.
4. Add secret/PII redaction and prompt-injection filtering before memory induction.
5. Build an automatic refresh worker that executes `refresh_status` candidates on a schedule.
6. Evaluate specialized and rapidly changing tool domains, including permission and backend-schema changes.
7. Add a baseline evaluation harness for success rate, incorrect-guidance rate, latency, calls, and lifecycle cost.
