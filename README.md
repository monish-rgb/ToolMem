# Mini ToolAtlas

A small, runnable implementation of the core ideas in **ToolAtlas: Learning Once, Reusing Everywhere with Tool-Side Memory** (arXiv:2607.11126).

This is intentionally a teaching prototype, not a reproduction of the paper's full benchmark. It keeps the important architecture:

- **Tool-Trace graph:** verified executions become agent-neutral `(tool, rationale)` traces.
- **Tool-Capability graph:** tools accumulate evidence-backed affordances, boundaries, and co-usage patterns.
- **Tool-Strategy graph:** repeated multi-tool sequences become reusable planning strategies.
- **Dynamic traversal:** a new task seeds retrieval from similar traces and follows graph links under a read budget.
- **Provider-side persistence:** memory is JSON owned by the MCP tool provider and can be reused by a different client or agent.

The expensive LLM proposer/reflection and embedding model from the research system are replaced with deterministic rules and local bag-of-words similarity. This keeps the end-to-end mechanism inspectable and API-key free.

## Run on Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\toolatlas-demo
```

The demo starts two real local **stdio MCP servers**:

1. `toolatlas-text-server` provides `normalize_text`, `word_count`, and `keyword_count`.
2. `toolatlas-memory-server` provides `register_tools`, `remember_execution`, `suggest_probes`, `get_guidance`, `inspect_tool`, and `memory_stats`.

It executes two successful composed tasks plus one failing boundary probe, verifies their outcomes, persists `demo-memory.json`, and retrieves guidance for an unseen task.

## Connect the servers to an MCP host

Use the virtual environment's absolute Python path in your host configuration:

```json
{
  "mcpServers": {
    "text-tools": {
      "command": "C:\\path\\to\\ToolMem\\.venv\\Scripts\\python.exe",
      "args": ["-m", "toolatlas.text_tools_server"]
    },
    "toolatlas-memory": {
      "command": "C:\\path\\to\\ToolMem\\.venv\\Scripts\\python.exe",
      "args": ["-m", "toolatlas.memory_server"],
      "env": {"TOOLATLAS_MEMORY_PATH": "C:\\path\\to\\memory.json"}
    }
  }
}
```

Recommended agent flow:

1. Call `get_guidance` before solving a task and place the returned playbook/tips in the agent context.
2. Use the ordinary provider tools.
3. Verify the result externally.
4. Call `remember_execution` with only agent-neutral rationales and the verified outcome.

## Where this differs from the full paper

The paper uses three seed tasks per tool, four rollouts per task, three exploration rounds with three boundary and three affordance probes per target, semantic embeddings, and an LLM navigator with an eight-read budget. This prototype supports the same data flow and the default `top_k=3`, `read_budget=8`, but the included demo is deliberately smaller. See `src/toolatlas/memory.py` for the induction/traversal logic and `src/toolatlas/demo.py` for the execution-verified MCP loop.
