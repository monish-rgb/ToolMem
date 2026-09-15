# Mini ToolAtlas

A small, runnable implementation of the core ideas in **ToolAtlas: Learning Once, Reusing Everywhere with Tool-Side Memory** (arXiv:2607.11126).

This is intentionally a teaching prototype, not a reproduction of the paper's full benchmark. It keeps the important architecture:

- **Tool-Trace graph:** verified executions become agent-neutral `(tool, rationale)` traces.
- **Tool-Capability graph:** tools accumulate evidence-backed affordances, boundaries, and co-usage patterns.
- **Tool-Strategy graph:** repeated multi-tool sequences become reusable planning strategies.
- **Dynamic traversal:** a new task seeds retrieval from similar traces and follows graph links under a read budget.
- **Provider-side persistence:** memory is stored in a SQLite/WAL database owned by the MCP tool provider and reused by different clients or agents.
- **Lifecycle management:** schema fingerprints, verification age, confidence, provenance, refresh candidates, and governance status prevent outdated memory from being served silently.

The expensive LLM proposer/reflection and embedding model from the research system are replaced with deterministic rules and local bag-of-words similarity. This keeps the end-to-end mechanism inspectable and API-key free.

## Run on Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\toolatlas-demo
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

## Connect the servers to an MCP host

The repository now includes two ready project configurations:

- `.mcp.json` for clients that support the common project MCP format and launch from the repository root.
- `.vscode/mcp.json` for VS Code with `${workspaceFolder}` paths.

Both configurations expose the installed official Filesystem server and ToolAtlas memory server. The toy text server is retained only for the small demo and unit tests.

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
