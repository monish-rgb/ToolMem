# ToolAtlas MCPMark integration (Phase 2)

Thin integration of ToolAtlas provider-side memory with the pinned official
MCPMark runner (`cd45b7f57923b9b3985467f5139927575f83141c`). Both arms use the
same `ToolAtlasMCPMarkAgent` code path; the mode flag selects behavior.

## Layout

- `integration/toolatlas_mcpmark/` — the integration package (copied verbatim
  into the checkout by the apply script):
  - `agent.py` — `ToolAtlasAgentMixin` + `create_agent_class()` binding it to
    the real `MCPMarkAgent`. Interception points reuse official execution:
    `_create_mcp_server` (trace wrapper), `_execute_litellm_with_tools` and
    `_execute_claude_native_with_tools` (guidance/rollout capture, signatures
    identical). No model or provider-tool logic is copied.
  - `tracing.py` — `TracedMCPServer` instruments the MCP boundary: every
    attempted call is recorded before awaiting the provider and updated with
    its result or exception, so raised calls are counted with their error.
    `list_tools` is a discovery event, kept separate from tool invocations.
  - `sanitize.py` — agent-neutral, environment-invariant evidence: strips
    secrets, paths, quoted literals, numbers; fixed guidance template; empty
    retrieval renders nothing.
  - `memory_stdio.py` — real stdio memory sessions with a sanitized child env
    (database path + read-only flag only; credentials never inherited).
  - `wiring.py` — registry/CLI/evaluator helpers used by the patch.
- `integration/tests/test_wiring.py` — executed **inside the pinned image**:
  registry, CLI flags, evaluator option forwarding, and a deterministic
  off-equivalence loop test (stubbed model + fake provider server).
- `patches/0001-toolatlas-agent.patch` — the complete pinned-checkout diff,
  generated from a clean apply. Verify with `git apply --check`.
- `scripts/apply_integration.py` — copies the package + applies anchor edits.
- `scripts/freeze_memory.py` — see `src/toolatlas/freeze_memory.py`
  (checkpoint/truncate WAL → SQLite backup API → integrity_check → hashes).
- `manifests/` — pre-registered split + paired schedule (Phase 3).
- `schemas/` — reserved for manifest/attempt JSON schemas (Phase 3).

## Modes

| Mode | Memory calls | Writes | Prompt |
|---|---|---|---|
| `off` | 0 | impossible | identical to stock |
| `read` | exactly 1 `get_guidance` (read-only server) | impossible | + fixed block only if retrieval non-empty |
| `learn` | 0 during execution; post-verifier ingestion only | verified rollouts only | identical to stock |

## Commands

```powershell
# Apply to a pristine pinned checkout copy (fails loudly on anchor drift)
.\.venv\Scripts\python benchmarks/mcpmark/scripts/apply_integration.py --checkout PATH

# ToolMem unit tests (fake base/provider/memory — no checkout or model needed)
.\.venv\Scripts\python -m pytest tests/test_toolatlas_mcpmark_agent.py tests/test_memory_freeze.py -q

# In-image wiring tests (pinned image + patched checkout mounted at /app)
docker run --rm -v <checkout>:/app -v <repo>/benchmarks/mcpmark/integration/tests:/wtests `
  mcpmark-pinned:cd45b7f sh -c "pip install -q pytest; python3 -m pytest /wtests/test_wiring.py -q"

# Freeze a training database (ToolMem venv)
.\.venv\Scripts\python -m toolatlas.freeze_memory --source training.db --output frozen.db
```

## Phase 2 gates (status)

- `toolatlas --mode off` matches the official agent loop (deterministic
  in-image test: same instruction, tools, temperature, max_tokens) — PASS.
- Raised provider calls counted with errors (ToolMem + trace tests) — PASS.
- Evaluation modes cannot invoke memory writes (read-only server profile +
  tool-list absence + error-result tests) — PASS.
- New files: 876 insertions, 1 deletion across 9 checkout files.
