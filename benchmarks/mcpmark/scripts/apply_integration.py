"""Apply the ToolAtlas integration to a pristine pinned MCPMark checkout.

Copies ``integration/toolatlas_mcpmark`` to ``src/toolatlas_mcpmark`` inside
the checkout, then applies small anchor-based edits (registry, CLI flags,
evaluator plumbing). Anchors are exact ASCII matches; the script fails loudly
on missing or ambiguous anchors instead of guessing.

Usage:
    python benchmarks/mcpmark/scripts/apply_integration.py --checkout PATH
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve()
INTEGRATION_SRC = HERE.parent.parent / "integration" / "toolatlas_mcpmark"


@dataclass
class Edit:
    path: str
    old: str
    new: str
    count: int = 1  # expected occurrences; use -1 for replace-all


EDITS = [
    Edit(
        "src/agents/__init__.py",
        "from .base_agent import BaseMCPAgent\n"
        "from .mcpmark_agent import MCPMarkAgent\n"
        "from .react_agent import ReActAgent\n"
        "\n"
        "AGENT_REGISTRY = {\n"
        '    "mcpmark": MCPMarkAgent,\n'
        '    "react": ReActAgent,\n'
        "}\n"
        "\n"
        '__all__ = ["BaseMCPAgent", "MCPMarkAgent", "ReActAgent", "AGENT_REGISTRY"]\n',
        "from .base_agent import BaseMCPAgent\n"
        "from .mcpmark_agent import MCPMarkAgent\n"
        "from .react_agent import ReActAgent\n"
        "from src.toolatlas_mcpmark.agent import create_agent_class\n"
        "\n"
        "ToolAtlasMCPMarkAgent = create_agent_class(MCPMarkAgent)\n"
        "\n"
        "AGENT_REGISTRY = {\n"
        '    "mcpmark": MCPMarkAgent,\n'
        '    "react": ReActAgent,\n'
        '    "toolatlas": ToolAtlasMCPMarkAgent,\n'
        "}\n"
        "\n"
        '__all__ = [\n'
        '    "BaseMCPAgent",\n'
        '    "MCPMarkAgent",\n'
        '    "ReActAgent",\n'
        '    "ToolAtlasMCPMarkAgent",\n'
        '    "AGENT_REGISTRY",\n'
        "]\n",
    ),
    Edit(
        "pipeline.py",
        "from src.model_config import ModelConfig\n",
        "from src.model_config import ModelConfig\n"
        "from src.toolatlas_mcpmark.wiring import (\n"
        "    add_toolatlas_cli_args,\n"
        "    toolatlas_options_from_args,\n"
        ")\n",
    ),
    Edit(
        "pipeline.py",
        '        help="Reasoning effort level for supported models (default: None)",\n'
        "    )\n"
        "\n"
        "    # Output configuration\n",
        '        help="Reasoning effort level for supported models (default: None)",\n'
        "    )\n"
        "\n"
        "    add_toolatlas_cli_args(parser)\n"
        "\n"
        "    # Output configuration\n",
    ),
    Edit(
        "pipeline.py",
        "                compaction_token=args.compaction_token,\n"
        "            )\n",
        "                compaction_token=args.compaction_token,\n"
        "                **toolatlas_options_from_args(args),\n"
        "            )\n",
    ),
    Edit(
        "src/evaluator.py",
        "from src.agents import AGENT_REGISTRY\n",
        "from src.agents import AGENT_REGISTRY\n"
        "from src.toolatlas_mcpmark.wiring import (\n"
        "    maybe_begin_attempt,\n"
        "    maybe_configure_agent,\n"
        "    maybe_run_learning_hook,\n"
        ")\n",
    ),
    Edit(
        "src/evaluator.py",
        '        agent_name: str = "mcpmark",\n'
        '        task_suite: str = "standard",\n'
        "        compaction_token: int = 0,\n"
        "    ):\n",
        '        agent_name: str = "mcpmark",\n'
        '        task_suite: str = "standard",\n'
        "        compaction_token: int = 0,\n"
        "        toolatlas_mode: str = \"off\",\n"
        "        toolatlas_memory: str | None = None,\n"
        "        toolatlas_trace_dir: str | None = None,\n"
        "        toolatlas_top_k: int = 3,\n"
        "        toolatlas_read_budget: int = 8,\n"
        "        toolatlas_token_budget: int = 384,\n"
        "        toolatlas_embed_mode: str = \"lexical\",\n"
        "    ):\n",
    ),
    Edit(
        "src/evaluator.py",
        "            reasoning_effort=self.reasoning_effort,\n"
        "            compaction_token=compaction_token,\n"
        "        )\n",
        "            reasoning_effort=self.reasoning_effort,\n"
        "            compaction_token=compaction_token,\n"
        "        )\n"
        "        self._toolatlas_options = {\n"
        '            "mode": toolatlas_mode,\n'
        '            "memory_path": toolatlas_memory,\n'
        '            "trace_dir": toolatlas_trace_dir,\n'
        '            "top_k": toolatlas_top_k,\n'
        '            "read_budget": toolatlas_read_budget,\n'
        '            "token_budget": toolatlas_token_budget,\n'
        '            "embed_mode": toolatlas_embed_mode,\n'
        "        }\n"
        "        maybe_configure_agent(self.agent, self._toolatlas_options)\n",
    ),
    Edit(
        "src/evaluator.py",
        "        task_instruction = self.task_manager.get_task_instruction(task)\n",
        "        task_instruction = self.task_manager.get_task_instruction(task)\n"
        '        task_ref = "%s/%s" % (\n'
        '            getattr(task, "category_id", ""),\n'
        '            getattr(task, "task_id", ""),\n'
        "        )\n"
        "        maybe_begin_attempt(self.agent, task_ref)\n",
    ),
    Edit(
        "src/evaluator.py",
        "        verify_time = time.time() - verify_start_time\n",
        "        toolatlas_hook = maybe_run_learning_hook(\n"
        "            self.agent,\n"
        "            task_id=task_ref,\n"
        "            instruction=task_instruction,\n"
        '            verifier_success=bool(getattr(result, "success", False)),\n'
        '            verification_output=str(getattr(result, "verification_output", "") or ""),\n'
        "        )\n"
        "        try:\n"
        '            with open(task_output_dir / "toolatlas_hook.json", "w", encoding="utf-8") as _hook_file:\n'
        "                json.dump(toolatlas_hook, _hook_file, indent=2, default=str)\n"
        "        except Exception:\n"
        "            pass\n"
        "        verify_time = time.time() - verify_start_time\n",
    ),
    Edit(
        "src/evaluator.py",
        '                "timeout": self.timeout,\n'
        '                "agent_name": self.agent_name,\n',
        '                "timeout": self.timeout,\n'
        '                "agent_name": self.agent_name,\n'
        '                "toolatlas_mode": self._toolatlas_options.get("mode", "off"),\n'
        '                "toolatlas_memory": self._toolatlas_options.get("memory_path"),\n'
        '                "toolatlas_trace_dir": self._toolatlas_options.get("trace_dir"),\n'
        '                "toolatlas_top_k": self._toolatlas_options.get("top_k"),\n'
        '                "toolatlas_read_budget": self._toolatlas_options.get("read_budget"),\n'
        '                "toolatlas_token_budget": self._toolatlas_options.get("token_budget"),\n'
        '                "toolatlas_embed_mode": self._toolatlas_options.get("embed_mode"),\n',
        count=-1,
    ),
]


def apply_to_checkout(checkout: Path) -> list[str]:
    if not (checkout / "pipeline.py").exists():
        raise FileNotFoundError(f"not an MCPMark checkout: {checkout}")
    package_src = INTEGRATION_SRC
    if not package_src.is_dir():
        raise FileNotFoundError(f"integration package missing: {package_src}")
    dest = checkout / "src" / "toolatlas_mcpmark"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(package_src, dest)
    applied = [f"copied integration package to src/toolatlas_mcpmark ({len(list(dest.rglob('*.py')))} files)"]
    for edit in EDITS:
        target = checkout / edit.path
        text = target.read_text(encoding="utf-8")
        found = text.count(edit.old)
        if edit.count == -1:
            if found < 1:
                raise RuntimeError(f"anchor not found in {edit.path}: {edit.old[:80]!r}")
            text = text.replace(edit.old, edit.new)
            applied.append(f"{edit.path}: replaced {found} occurrence(s)")
        else:
            if found != edit.count:
                raise RuntimeError(
                    f"anchor in {edit.path} found {found}x, expected {edit.count}x: {edit.old[:80]!r}"
                )
            text = text.replace(edit.old, edit.new, edit.count)
            applied.append(f"{edit.path}: applied edit")
        target.write_text(text, encoding="utf-8")
    return applied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply the ToolAtlas integration.")
    parser.add_argument("--checkout", required=True, help="Pinned MCPMark checkout path")
    args = parser.parse_args(argv)
    try:
        for line in apply_to_checkout(Path(args.checkout)):
            print(line)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
