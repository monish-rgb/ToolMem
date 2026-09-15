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
- `src/toolatlas/memory_server.py`: `create_memory_server(path)` factory + module-level `mcp` reading `TOOLATLAS_MEMORY_PATH` (default `.toolatlas/memory.json`). Tools: `register_tools`, `remember_execution`, `get_guidance(top_k=3, read_budget=8)`, `inspect_tool`, `suggest_probes`, `memory_stats`; resource `toolatlas://graph`.
- `src/toolatlas/memory.py`: `ToolMemory.induce()` / `guide()` / `suggest_probes()` / `save()` (atomic tmp+replace). Backbone = shortest successful rollout.
- `src/toolatlas/similarity.py`: lexical `tokens()` + `cosine_text()` — no synonym handling.
- `src/toolatlas/demo.py`: real stdio subprocess flow via `mcp.Client` + `StdioServerParameters`: list → `register_tools` → 2 verified `normalize_text→keyword_count` learns → 1 empty-keyword probe → `get_guidance`. Writes `demo-memory.json`.
- `tests/test_memory.py`: pure `ToolMemory` persistence/traversal. `tests/test_mcp.py`: in-process `Client(server)` (no subprocess, unlike demo).

## Rules that are easy to break

- Keep MCP stdio end-to-end; do not collapse demo/tests to direct Python calls.
- Demo stderr `Tool 'keyword_count' failed: ... keyword must not be empty` is expected. A Python traceback is not.
- `demo-memory.json`, `.toolatlas/`, `paper.*` are gitignored; demo overwrites `demo-memory.json`.
- `remember_execution` only with verified `resolved` + agent-neutral rationale (intent, not chain-of-thought). Entries must stay environment-invariant: no secrets, PII, literal user data, paths, or agent syntax — `_generic_rationale()` strips paths/quotes/numbers.
- Capability dedup (`_merge_entry`) is exact-text match; each entry must keep `source_queries`. Failed rollouts become `boundaries` cautions, not affordances.
- `guide()` returns empty `seed_candidates`/`playbook` when lexical similarity is 0 — do not invent generic advice. Strategies form only when the same multi-tool sequence appears in ≥2 traces.
- Clean demo stats: `tools: 3, traces: 3, strategies: 1, trace_edges: 3`.

## Verify

Rerun `pytest -q` after touching `memory.py`, `models.py`, either server, or `similarity.py`; run `toolatlas-demo` after changing the MCP loop or persistence.
