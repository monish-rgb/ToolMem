"""Offline paper-pipeline demo: LLM-proposed seeds, reflected induction, hybrid retrieval.

Runs entirely offline with canned LLM responses (FakeLLMCall) and the
deterministic hash embedder: no credentials, no network, no paid calls.
Pass ``--live`` with explicitly exported credentials to use a real model
for proposal and reflection instead.

Usage:
    python -m toolatlas.llm_memory_demo
    python -m toolatlas.llm_memory_demo --live   # needs $env LLM key + model
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .embeddings import HashEmbedder
from .llm_client import FakeLLMCall
from .memory import ToolMemory
from .models import ToolSpec
from .offline_builder import OfflineCost, build_llm_memory

SEED_JSON = json.dumps({"tasks": [
    {"summary": "Tally tiny files into groups for review",
     "steps": [{"tool": "list_files", "goal": "establish the file set"},
               {"tool": "move_files", "goal": "file each item by size"}]},
    {"summary": "Weigh tiny files before archiving",
     "steps": [{"tool": "list_files", "goal": "inspect the file set"},
               {"tool": "move_files", "goal": "archive each item by weight"}]},
    {"summary": "Grade tiny files into tiers",
     "steps": [{"tool": "list_files", "goal": "scope the file set"},
               {"tool": "move_files", "goal": "grade each item into tiers"}]},
]})

REFLECT_JSON = json.dumps({
    "rationales": ["establish working scope and inputs",
                   "produce and verify the task output"],
    "tips": ["Verify the final tool result against the task requirement."],
})


def _executor(task_id: str, planned: list[dict]) -> list[dict]:
    return [{"tool": step["tool"], "rationale": step.get("goal") or "do it"}
            for step in planned]


def _verifier(task_id: str, steps: list[dict]) -> tuple[bool, str]:
    return True, "offline demo verifier accepted the organized listing"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline LLM-memory pipeline demo.")
    parser.add_argument("--live", action="store_true",
                        help="use a real model via explicitly exported credentials")
    args = parser.parse_args(argv)

    if args.live:
        from .llm_client import llm_call_from_env
        try:
            llm: object = llm_call_from_env()
        except RuntimeError as exc:
            print(f"live demo unavailable: {exc}")
            return 2
    else:
        llm = FakeLLMCall(
            responses={"TARGET TOOL: list_files": SEED_JSON,
                       "TARGET TOOL: move_files": SEED_JSON,
                       "TASK: Tally": REFLECT_JSON,
                       "TASK: Weigh": REFLECT_JSON,
                       "TASK: Grade": REFLECT_JSON},
            default=REFLECT_JSON)

    work = Path(tempfile.mkdtemp(prefix="llm-memory-demo"))
    memory = ToolMemory(work / "demo.db")
    tools = ["list_files", "move_files"]
    memory.register_tools([ToolSpec(name=name, description=f"demo tool {name}")
                           for name in tools])
    cost = OfflineCost()
    report = build_llm_memory(memory, tools, _executor, _verifier, llm,  # type: ignore[arg-type]
                              HashEmbedder(), cost=cost)
    print(json.dumps({"stats": memory.stats(), "build": report}, indent=2))

    query = "Compute little documents for review"  # paraphrase, zero lexical overlap
    lexical = memory.guide(query)
    hybrid = memory.guide(query, embed_mode="embedding", embedder=HashEmbedder())
    print(json.dumps({
        "query": query,
        "lexical_seeds": [c["qid"] for c in lexical["seed_candidates"]],
        "lexical_retrieval": lexical["traversal"]["retrieval"],
        "hybrid_seeds": [c["qid"] for c in hybrid["seed_candidates"]],
        "hybrid_retrieval": hybrid["traversal"]["retrieval"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
