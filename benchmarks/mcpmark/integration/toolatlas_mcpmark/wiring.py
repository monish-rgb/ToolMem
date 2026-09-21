"""Wiring helpers applied to the pinned MCPMark checkout by the patch.

Kept separate from the agent logic so the patch itself stays small: it only
imports this package, registers the agent, adds CLI flags, and forwards the
parsed options plus the post-verifier hook.
"""

from __future__ import annotations

from typing import Any

TOOLATLAS_DEFAULT_TOP_K = 3
TOOLATLAS_DEFAULT_READ_BUDGET = 8
TOOLATLAS_DEFAULT_TOKEN_BUDGET = 384
TOOLATLAS_DEFAULT_EMBED_MODE = "lexical"


def register_agent(registry: dict, agent_cls: Any) -> dict:
    """Register the ToolAtlas agent without disturbing existing entries."""
    registry = dict(registry)
    if "toolatlas" in registry:
        raise ValueError("AGENT_REGISTRY already contains 'toolatlas'")
    registry["toolatlas"] = agent_cls
    return registry


def add_toolatlas_cli_args(parser: Any) -> Any:
    """Add explicit, provenance-visible ToolAtlas options to the pipeline."""
    group = parser.add_argument_group("ToolAtlas memory")
    group.add_argument(
        "--toolatlas-mode",
        default="off",
        choices=["off", "learn", "read"],
        help="ToolAtlas memory mode (default: off)",
    )
    group.add_argument(
        "--toolatlas-memory",
        default=None,
        help="Path to the ToolAtlas memory database",
    )
    group.add_argument(
        "--toolatlas-trace-dir",
        default=None,
        help="Directory for per-attempt MCP trace records",
    )
    group.add_argument(
        "--toolatlas-top-k",
        type=int,
        default=TOOLATLAS_DEFAULT_TOP_K,
        help="Guidance seed candidates (default: 3)",
    )
    group.add_argument(
        "--toolatlas-read-budget",
        type=int,
        default=TOOLATLAS_DEFAULT_READ_BUDGET,
        help="Guidance traversal read budget (default: 8)",
    )
    group.add_argument(
        "--toolatlas-token-budget",
        type=int,
        default=TOOLATLAS_DEFAULT_TOKEN_BUDGET,
        help="Hard guidance token budget for the injected block (default: 384)",
    )
    group.add_argument(
        "--toolatlas-embed-mode",
        default=TOOLATLAS_DEFAULT_EMBED_MODE,
        choices=["lexical", "embedding"],
        help="Guidance seed retrieval: lexical or embedding blend (default: lexical)",
    )
    return parser


def toolatlas_options_from_args(args: Any) -> dict:
    """Extract ToolAtlas options with safe defaults for unpatched arg sets."""
    return {
        "mode": getattr(args, "toolatlas_mode", "off"),
        "memory_path": getattr(args, "toolatlas_memory", None),
        "trace_dir": getattr(args, "toolatlas_trace_dir", None),
        "top_k": getattr(args, "toolatlas_top_k", TOOLATLAS_DEFAULT_TOP_K),
        "read_budget": getattr(args, "toolatlas_read_budget", TOOLATLAS_DEFAULT_READ_BUDGET),
        "token_budget": getattr(args, "toolatlas_token_budget", TOOLATLAS_DEFAULT_TOKEN_BUDGET),
        "embed_mode": getattr(args, "toolatlas_embed_mode", TOOLATLAS_DEFAULT_EMBED_MODE),
    }


def maybe_configure_agent(agent: Any, options: dict) -> None:
    """Configure a ToolAtlas agent; no-op for stock agents."""
    configure = getattr(agent, "configure_toolatlas", None)
    if configure is None:
        return
    configure(
        mode=options.get("mode", "off"),
        memory_path=options.get("memory_path"),
        trace_dir=options.get("trace_dir"),
        top_k=options.get("top_k", TOOLATLAS_DEFAULT_TOP_K),
        read_budget=options.get("read_budget", TOOLATLAS_DEFAULT_READ_BUDGET),
        token_budget=options.get("token_budget", TOOLATLAS_DEFAULT_TOKEN_BUDGET),
        embed_mode=options.get("embed_mode", TOOLATLAS_DEFAULT_EMBED_MODE),
    )


def maybe_begin_attempt(agent: Any, task_id: str, arm: str = "", attempt: int = 0) -> None:
    """Give the agent per-attempt context; no-op for stock agents."""
    begin = getattr(agent, "begin_attempt", None)
    if begin is None:
        return
    begin(task_id, arm, attempt)


def maybe_run_learning_hook(
    agent: Any,
    *,
    task_id: str,
    instruction: str,
    verifier_success: bool,
    verification_output: str,
) -> dict:
    """Run the post-verifier hook; never affects verification or cleanup."""
    hook = getattr(agent, "on_verified_rollout", None)
    if hook is None:
        return {"ingested": False, "reason": "stock agent has no learning hook"}
    try:
        return hook(
            task_id=task_id,
            instruction=instruction,
            verifier_success=verifier_success,
            verification_output=verification_output,
        ) or {"ingested": False, "reason": "hook returned nothing"}
    except Exception as exc:  # hook failures must not break the run
        return {"ingested": False, "reason": f"hook error: {type(exc).__name__}: {exc}"}
