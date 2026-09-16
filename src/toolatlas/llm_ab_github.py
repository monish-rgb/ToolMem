from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from .github_demo import (
    GITHUB_PACKAGE_VERSION,
    GITHUB_PROVIDER,
    READ_ONLY_GITHUB_TOOLS,
    _test_repo,
    _value,
    github_server_path,
)
from .llm_ab_nim import DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_TEMPERATURE, _openai_client, _parse_answer_int
from .memory_server import create_memory_server

DEFAULT_MAX_STEPS = 8

SYSTEM_PROMPT = (
    "You are a GitHub MCP agent. Use only the provided tools. "
    "Answer with the exact requested integer."
)
# Neutral baseline: no tool-sequence hint. Arm B gets it via memory.
# Counts are resolved at runtime (repos change), so expected=None means
# "verify against an independent direct read after the run".
TASKS = [
    {"kind": "issue_count", "task": "List the open issues for triage review and report how many were returned",
     "required": ["list_issues"]},
    {"kind": "commit_count", "task": "Review the recent commits for context and report how many were returned",
     "required": ["list_commits"]},
    {"kind": "overview", "task": "Inspect the open issues and recent commits for triage and report the total items returned",
     "required": ["list_issues", "list_commits"]},
]


@dataclass
class GitHubAudit:
    client: Client
    calls: list[dict[str, Any]] = field(default_factory=list)
    forbidden_attempts: int = 0
    payloads: list[str] = field(default_factory=list)

    async def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        if tool not in READ_ONLY_GITHUB_TOOLS:
            self.forbidden_attempts += 1
            raise PermissionError(f"write-capable GitHub tool blocked: {tool}")
        started = time.perf_counter()
        try:
            result = await self.client.call_tool(tool, arguments)
        except Exception as exc:
            self.calls.append({"tool": tool, "is_error": True, "latency_ms": round((time.perf_counter() - started) * 1000, 3)})
            self.payloads.append(f"ERROR: {exc}")
            raise
        self.calls.append(
            {"tool": tool, "is_error": result.is_error, "latency_ms": round((time.perf_counter() - started) * 1000, 3)}
        )
        text = str(_value(result))
        self.payloads.append(text)
        return result


def mcp_to_openai_tools(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {
            "name": t.name, "description": t.description or "",
            "parameters": t.input_schema or {"type": "object", "properties": {}}}}
        for t in mcp_tools if t.name in READ_ONLY_GITHUB_TOOLS
    ]


async def _seed_training_memory(audit: GitHubAudit, memory: Client, owner: str, repo: str) -> None:
    probe = await audit.call("list_commits", {"owner": owner, "repo": repo})
    await memory.call_tool("remember_execution", {
        "task_id": "github_repo_access", "summary": "Review recent commits for context",
        "steps": [{"tool": "list_commits", "rationale": "review recent commits for context"}],
        "resolved": not probe.is_error, "observation": "the commit history was retrieved",
        "verifier_type": "list_succeeded"})
    if probe.is_error:
        raise AssertionError("training verifier failed: check PAT scopes and GITHUB_TEST_REPOSITORY")
    for idx in (1, 2):
        issues = await audit.call("list_issues", {"owner": owner, "repo": repo, "state": "open", "perPage": 5})
        commits = await audit.call("list_commits", {"owner": owner, "repo": repo})
        ok = not issues.is_error and not commits.is_error
        await memory.call_tool("remember_execution", {
            "task_id": f"github_repo_overview_{idx}",
            "summary": "Inspect repository issues and recent commits for triage",
            "steps": [{"tool": "list_issues", "rationale": "list open issues for triage review"},
                      {"tool": "list_commits", "rationale": "review recent commits for context"}],
            "resolved": ok, "observation": "the issue list and commit history were both retrieved",
            "verifier_type": "both_tools_succeeded"})
        if not ok:
            raise AssertionError("training verifier failed for overview")


def _count_items(text: str) -> int | None:
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return len(data)
    except (json.JSONDecodeError, TypeError):
        pass
    return None


async def _independent_count(audit: GitHubAudit, kind: str, owner: str, repo: str) -> int | None:
    try:
        if kind == "issue_count":
            res = await audit.call("list_issues", {"owner": owner, "repo": repo, "state": "open", "perPage": 5})
            return _count_items(str(_value(res)))
        if kind == "commit_count":
            res = await audit.call("list_commits", {"owner": owner, "repo": repo})
            return _count_items(str(_value(res)))
        if kind == "overview":
            issues = await audit.call("list_issues", {"owner": owner, "repo": repo, "state": "open", "perPage": 5})
            commits = await audit.call("list_commits", {"owner": owner, "repo": repo})
            a, b = _count_items(str(_value(issues))), _count_items(str(_value(commits)))
            return (a or 0) + (b or 0) if a is not None and b is not None else None
    except Exception:
        return None
    return None


async def _run_llm_agent(model, temperature, max_steps, system_prompt, openai_tools, audit, owner, repo, task):
    client = _openai_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{task['task']}. Repository: {owner}/{repo} (pass as owner/repo args). Report only the integer."},
    ]
    steps = 0
    while steps < max_steps:
        steps += 1
        response = await asyncio.to_thread(
            client.chat.completions.create, model=model, messages=messages,
            tools=openai_tools or None, tool_choice="auto" if openai_tools else "none",
            temperature=temperature, max_tokens=512)
        choice = response.choices[0].message
        if not choice.tool_calls:
            text = choice.content or ""
            return {"final_text": text, "answer": _parse_answer_int(text), "steps": steps}
        messages.append({"role": "assistant", "content": choice.content, "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in choice.tool_calls]})
        for tc in choice.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                result = await audit.call(tc.function.name, args)
                payload = str(_value(result))
                if result.is_error:
                    payload = f"MCP tool error: {payload}"
            except PermissionError as exc:
                payload = f"Blocked: {exc}"
            except Exception as exc:
                payload = f"MCP tool error: {exc}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": payload[:8000]})
    return {"final_text": "", "answer": None, "steps": steps}


async def run_github_nim_ab(project_root, memory_path, owner, repo,
                             model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE, max_steps=DEFAULT_MAX_STEPS):
    project_root = Path(project_root).resolve()
    memory_path = Path(memory_path).resolve()
    server_js = github_server_path(project_root)
    if not server_js.is_file():
        raise FileNotFoundError("run npm install before the GitHub NIM A/B test")
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js is required")
    token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    if not token:
        raise RuntimeError("set $env:GITHUB_PERSONAL_ACCESS_TOKEN before running")
    params = StdioServerParameters(
        command=node, args=[str(server_js)],
        env={"PATH": os.environ.get("PATH", ""), "GITHUB_PERSONAL_ACCESS_TOKEN": token})
    memory_server = create_memory_server(memory_path)

    async with Client(params) as github, Client(memory_server) as memory:
        listed = await github.list_tools()
        await memory.call_tool("register_tools", {"tools": [
            {"name": t.name, "description": t.description or "", "input_schema": t.input_schema,
             "provider": GITHUB_PROVIDER, "version": GITHUB_PACKAGE_VERSION}
            for t in listed.tools if t.name in READ_ONLY_GITHUB_TOOLS]})
        openai_tools = mcp_to_openai_tools(listed.tools)
        training_audit = GitHubAudit(github)
        await _seed_training_memory(training_audit, memory, owner, repo)

        per_task, base_calls = [], 0
        assisted_calls = base_pass = assisted_pass = 0
        last_guidance: dict[str, Any] = {}
        for task in TASKS:
            verify_audit = GitHubAudit(github)
            expected = await _independent_count(verify_audit, task["kind"], owner, repo)

            base_audit = GitHubAudit(github)
            started = time.perf_counter()
            base = await _run_llm_agent(model, temperature, max_steps, SYSTEM_PROMPT, openai_tools, base_audit, owner, repo, task)
            base_elapsed = time.perf_counter() - started

            guidance = _value(await memory.call_tool(
                "get_guidance", {"task": "Inspect repository issues and recent commits for triage", "top_k": 3, "read_budget": 8}))
            last_guidance = guidance
            assisted_prompt = SYSTEM_PROMPT + f"\nLearned playbook: {json.dumps(guidance.get('playbook', []))}"
            assisted_audit = GitHubAudit(github)
            started = time.perf_counter()
            assisted = await _run_llm_agent(model, temperature, max_steps, assisted_prompt, openai_tools, assisted_audit, owner, repo, task)
            assisted_elapsed = time.perf_counter() - started

            def metrics(name, run, audit, elapsed):
                tools_used = [c["tool"] for c in audit.calls]
                count_ok = (expected is None) or (run["answer"] == expected)
                req_ok = all(r in tools_used for r in task["required"])
                return {"agent": name, "task": task["task"], "passed": bool(count_ok and req_ok),
                        "answer": run["answer"], "expected": expected,
                        "required_tools_used": req_ok, "final_text": run["final_text"][:300],
                        "llm_steps": run["steps"], "tool_calls": len(audit.calls),
                        "tools_used": tools_used, "forbidden_write_attempts": audit.forbidden_attempts,
                        "elapsed_ms": round(elapsed * 1000, 3)}

            base_m = metrics("baseline_without_toolatlas", base, base_audit, base_elapsed)
            assisted_m = metrics("agent_with_toolatlas", assisted, assisted_audit, assisted_elapsed)
            base_calls += base_m["tool_calls"]
            assisted_calls += assisted_m["tool_calls"]
            base_pass += int(base_m["passed"])
            assisted_pass += int(assisted_m["passed"])
            per_task.append({"baseline": base_m, "toolatlas": assisted_m,
                             "verify_calls_excluded": len(verify_audit.calls)})

    return {"benchmark": "github_nim_llm_ab", "model": model, "temperature": temperature,
            "verifier": "model_count_matches_independent_read_and_required_tools_used",
            "github_server": f"{GITHUB_PROVIDER}@{GITHUB_PACKAGE_VERSION}", "test_repository": f"{owner}/{repo}",
            "training_calls_excluded": len(training_audit.calls), "per_task": per_task,
            "totals": {"baseline_calls": base_calls, "toolatlas_calls": assisted_calls,
                       "calls_saved": base_calls - assisted_calls,
                       "baseline_passed": f"{base_pass}/{len(TASKS)}", "toolatlas_passed": f"{assisted_pass}/{len(TASKS)}"},
            "retrieved_guidance": last_guidance}


def main() -> None:
    parser = argparse.ArgumentParser(description="True LLM A/B with NVIDIA NIM + live GitHub MCP (read-only)")
    parser.add_argument("--memory", type=Path, default=Path(".toolatlas/github-nim-benchmark.db"))
    parser.add_argument("--owner", default="")
    parser.add_argument("--repo", default="")
    parser.add_argument("--model", default=os.environ.get("NVIDIA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = parser.parse_args()
    owner, repo = (args.owner, args.repo) if args.owner and args.repo else _test_repo()
    result = asyncio.run(run_github_nim_ab(Path.cwd(), args.memory, owner, repo, args.model, args.temperature, args.max_steps))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
