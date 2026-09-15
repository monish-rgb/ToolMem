# ToolAtlas – Agent Guide

Minimal, local prototype of ToolAtlas provider-side memory (arXiv:2607.11126). Teaching demo, not a full paper reproduction: deterministic induction + lexical similarity instead of LLM proposer/reflection/embeddings.

## Commands (Windows PowerShell – always use `.venv`)

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m pytest tests/test_memory.py -q
.\.venv\Scripts\python -m compileall -q src tests
.\.venv\Scripts\toolatlas-demo
```

Recreate env: `python -m venv .venv; .\.venv\Scripts\python -m pip install -e ".[dev]"`. Requires Python >=3.11. Only runtime dep is `mcp>=2.0,<3` (`pyproject.toml`).

## Architecture

- `src/toolatlas/text_tools_server.py`: ordinary MCP tools `normalize_text`, `word_count`, `keyword_count`. Empty keyword raises `ToolError` — intentional boundary probe, not a bug.
- `src/toolatlas/filesystem_demo.py`: end-to-end integration with the pinned official Filesystem MCP server. It must remain restricted to `mcp-sandbox` (or a test temporary directory).
- `package.json` pins `@modelcontextprotocol/server-filesystem`; `.mcp.json` and `.vscode/mcp.json` configure Filesystem + ToolAtlas Memory as the repository defaults.
- `src/toolatlas/memory_server.py`: `create_memory_server(path)` factory + module-level `mcp` reading `TOOLATLAS_MEMORY_PATH` (default `.toolatlas/memory.db`). Tools cover single/batch ingestion, guidance, probes, refresh, re-verification, governance, inspection, and stats.
- `src/toolatlas/memory.py`: lifecycle-aware `ToolMemory`; shortest successful rollout is the trace backbone. Guidance includes provenance, confidence, expiry filtering, and an explicit traversal audit.
- `src/toolatlas/storage.py`: SQLite/WAL persistence. Use one writer service per database in production.
- `src/toolatlas/similarity.py`: lexical `tokens()` + `cosine_text()` — no synonym handling.
- `src/toolatlas/demo.py`: real stdio subprocess flow via `mcp.Client` + `StdioServerParameters`: list → `register_tools` → 2 verified `normalize_text→keyword_count` learns → 1 empty-keyword probe → `get_guidance`. Writes `demo-memory.db`.
- `tests/test_memory.py`: pure `ToolMemory` persistence/traversal. `tests/test_mcp.py`: in-process `Client(server)` (no subprocess, unlike demo).
- `tests/test_real_filesystem_mcp.py`: live stdio test of the locally installed official Filesystem server, including sandbox-denial verification.
- `src/toolatlas/readonly_benchmark.py` and `tests/test_readonly_ab.py`: live read-only deterministic A/B comparison. It must never call tools outside `READ_ONLY_TOOLS`.

## Rules that are easy to break

- Keep MCP stdio end-to-end; do not collapse demo/tests to direct Python calls.
- Do not broaden the Filesystem MCP allowlist beyond `mcp-sandbox` without explicit user approval.
- Demo stderr `Tool 'keyword_count' failed: ... keyword must not be empty` is expected. A Python traceback is not.
- `demo-memory.db*`, `.toolatlas/`, `paper.*` are gitignored.
- `remember_execution` only with verified `resolved` + agent-neutral rationale (intent, not chain-of-thought). Entries must stay environment-invariant: no secrets, PII, literal user data, paths, or agent syntax — `_generic_rationale()` strips paths/quotes/numbers.
- Capability dedup (`_merge_entry`) is exact-text match; each entry must keep `source_queries`. Failed rollouts become `boundaries` cautions, not affordances.
- `guide()` returns empty `seed_candidates`/`playbook` when lexical similarity is 0 — do not invent generic advice. Strategies form only when the same multi-tool sequence appears in ≥2 traces.
- Tool schema/version/provider changes invalidate dependent memory. Stale, invalid, quarantined, and verification-expired traces must not be served.
- Default verification age is 30 days. Refresh requires an external rerun followed by `reverify_trace`; never mark memory current without verification.
- `set_trace_status` requires a reason and cannot activate memory; only successful `reverify_trace` can reactivate it.
- Clean demo stats: `tools: 3, traces: 3, strategies: 1, trace_edges: 3, executions: 3, stale_traces: 0`.

## Verify

Rerun `pytest -q` after touching `memory.py`, `models.py`, either server, or `similarity.py`; run `toolatlas-demo` after changing the MCP loop or persistence.

Current full suite: 9 tests. `npm install` is required to execute rather than skip the real Filesystem integrations.
