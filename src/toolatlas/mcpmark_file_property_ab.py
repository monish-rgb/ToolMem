"""Real-LLM both-arms A/B over MCPMark's official file_property slice.

Pipeline per task (size_classification, time_classification):
  1. TRAIN (deterministic, verified): run a scripted policy on a training
     snapshot copy, check the official verifier, induce two same-sequence
     traces into ToolAtlas memory so a strategy forms. Then FREEZE the DB.
  2. EVAL (real LLM, k independent rollouts per arm): fresh snapshot copy per
     attempt. Arm A = filesystem tools only. Arm B = one get_guidance call
     against the frozen DB, then the same filesystem tools. No ingestion or
     re-verification during eval. Each attempt is checked by the official
     verify.py with FILESYSTEM_TEST_DIR pointing at the attempt root.

Writes are confined to disposable snapshot copies: the Filesystem MCP server
is rooted at the attempt directory for every run.

Snapshot integrity: file_property.zip is verified against the SHA-256 pinned
in benchmarks/docker/Dockerfile before use. mtimes are preserved on restore
(time_classification depends on baked 2025 modification dates; note the
server's `created` field reflects extraction time and is a trap — `modified`
is the solvable signal).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from .filesystem_demo import filesystem_server_path
from .llm_ab_nim import _openai_client
from .memory_server import create_memory_server
from .readonly_benchmark import _value
from .guidance_render import estimate_tokens as estimate_guidance_tokens
from .guidance_render import render_compact_block
from .result_limit import truncate_result
from .tool_filter import estimate_schema_tokens, filter_tools_by_playbook

SNAPSHOT_SHA256 = "99d5449cef45bfcda6e5260f6ef4cd356bdbae59818d37ffa840054cdee19ec4"
PROVIDER = "io.github.modelcontextprotocol/server-filesystem"
VERSION = "2026.8.31"

SYSTEM_PROMPT = (
    "You are a filesystem agent with MCP tools. Complete the user's task "
    "inside the workspace root using only the provided tools. When finished, "
    "reply with a short summary of what you did."
)

TRAIN_SUMMARIES = {
    "size_classification": "Classify files in a directory by size and move them into category folders",
    "time_classification": "Organize files in a directory by date into month/day folders with metadata files",
}


@dataclass
class CallLog:
    calls: list[str] = field(default_factory=list)
    failed_provider_calls: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_usage: list[dict[str, Any]] = field(default_factory=list)
    request_rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def failed_calls(self) -> int:
        return len(self.failed_provider_calls)


def _content(result: Any) -> str:
    blocks = getattr(result, "content", None) or []
    return "\n".join(str(getattr(block, "text", "")) for block in blocks)


def check_snapshot(snapshot: Path) -> None:
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    if digest != SNAPSHOT_SHA256:
        raise AssertionError(f"snapshot SHA-256 mismatch: {digest}")


def restore_snapshot(snapshot: Path, root: Path) -> Path:
    """Extract top-level file_property/* files, preserving mtimes (needed for time task)."""
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    with zipfile.ZipFile(snapshot) as archive:
        for item in archive.infolist():
            parts = Path(item.filename).parts
            if len(parts) != 2 or parts[0] != "file_property" or item.is_dir():
                continue
            target = root / parts[1]
            assert target.resolve().is_relative_to(root.resolve())
            with archive.open(item) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            mtime = time.mktime(datetime(*item.date_time).timetuple())
            os.utime(target, (mtime, mtime))
    return root


def run_verifier(verify_py: Path, root: Path) -> tuple[bool, str]:
    env = {**os.environ, "FILESYSTEM_TEST_DIR": str(root),
           "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [sys.executable, str(verify_py)], env=env,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    return proc.returncode == 0, ((proc.stdout or "") + (proc.stderr or ""))[-2000:]


def _parse_modified(info: str) -> datetime | None:
    match = re.search(r"^modified:\s*(.+)$", info, re.MULTILINE | re.IGNORECASE)
    if not match:
        return None
    text = re.sub(r"\s*\(.*\)\s*$", "", match.group(1).strip())
    for fmt in ("%a %b %d %Y %H:%M:%S GMT%z", "%a %b %d %Y %H:%M:%S %Z"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    months = {m: i + 1 for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
    match = re.search(r"([A-Z][a-z]{2})\s+(\d{1,2})\s+(\d{4})", text)
    if match:
        return datetime(int(match.group(3)), months[match.group(1)], int(match.group(2)))
    return None


async def _size_policy(client: Client, root: Path, log: CallLog) -> None:
    async def call(name: str, args: dict[str, Any]) -> str:
        result = await client.call_tool(name, args)
        if result.is_error:
            raise AssertionError(f"training policy failed at {name}")
        log.calls.append(name)
        return _content(result)

    listing = await call("list_directory", {"path": str(root)})
    files = [line.removeprefix("[FILE] ") for line in listing.splitlines() if line.startswith("[FILE] ")]
    for category in ("small_files", "medium_files", "large_files"):
        await call("create_directory", {"path": str(root / category)})
    for name in files:
        info = await call("get_file_info", {"path": str(root / name)})
        match = re.search(r"^size:\s*(\d+)", info, re.MULTILINE | re.IGNORECASE)
        size = int(match.group(1))
        category = "small_files" if size < 300 else "medium_files" if size <= 700 else "large_files"
        await call("move_file", {"source": str(root / name), "destination": str(root / category / name)})


async def _time_policy(client: Client, root: Path, log: CallLog) -> None:
    async def call(name: str, args: dict[str, Any]) -> str:
        result = await client.call_tool(name, args)
        if result.is_error:
            raise AssertionError(f"training policy failed at {name}")
        log.calls.append(name)
        return _content(result)

    listing = await call("list_directory", {"path": str(root)})
    files = [line.removeprefix("[FILE] ") for line in listing.splitlines()
             if line.startswith("[FILE] ") and not line.endswith(".DS_Store")]
    buckets: dict[str, list[tuple[str, datetime, str]]] = {}
    for name in files:
        info = await call("get_file_info", {"path": str(root / name)})
        stamp = _parse_modified(info)
        if stamp is None:
            raise AssertionError(f"could not parse modified date for {name}: {info[:200]}")
        buckets.setdefault(f"{stamp.month:02d}/{stamp.day:02d}", []).append((name, stamp, info))
    for bucket, items in sorted(buckets.items()):
        month, day = bucket.split("/")
        await call("create_directory", {"path": str(root / month)})
        await call("create_directory", {"path": str(root / month / day)})
        for name, _, _ in items:
            await call("move_file", {"source": str(root / name),
                                     "destination": str(root / month / day / name)})
        ordered = sorted(items, key=lambda item: item[1])
        oldest = f"{ordered[0][0]} {ordered[0][1].strftime('%a %b %d %Y')}"
        latest = f"{ordered[-1][0]} {ordered[-1][1].strftime('%a %b %d %Y')}"
        await call("write_file", {"path": str(root / month / day / "metadata_analyse.txt"),
                                  "content": f"{oldest}\n{latest}\n"})


POLICIES = {"size_classification": _size_policy, "time_classification": _time_policy}
POLICY_TOOLS = {
    "size_classification": ["list_directory", "create_directory", "get_file_info", "move_file"],
    "time_classification": ["list_directory", "get_file_info", "create_directory", "move_file", "write_file"],
}
POLICY_RATIONALES = {
    "list_directory": "inspect the workspace directory for task files",
    "create_directory": "create the required category directories",
    "get_file_info": "read file metadata to determine the target category",
    "move_file": "move each file into its category directory",
    "write_file": "write the required per-directory metadata file",
}


async def train_task(task: str, snapshot: Path, workdir: Path, memory: Client) -> dict[str, Any]:
    """Run the deterministic policy twice on training copies; verify; induce strategy."""
    for run in (1, 2):
        root = restore_snapshot(snapshot, workdir / f"train-{task}-{run}")
        node = shutil.which("node")
        from .filesystem_demo import filesystem_server_path as _fsp
        params = StdioServerParameters(
            command=node, args=[str(_fsp(Path.cwd()).resolve()), str(root)])
        log = CallLog()
        async with Client(params) as client:
            await POLICIES[task](client, root, log)
        task_dir = TASKS_DIR / task
        passed, output = run_verifier(task_dir / "verify.py", root)
        await memory.call_tool("remember_execution", {
            "task_id": f"mcpmark_{task}_{run}", "summary": TRAIN_SUMMARIES[task],
            "steps": [{"tool": tool, "rationale": POLICY_RATIONALES[tool]} for tool in POLICY_TOOLS[task]],
            "resolved": passed, "observation": "the official verifier accepted the organized directory",
            "verifier_type": "official_mcpmark_verifier"})
        if not passed:
            raise AssertionError(f"training policy failed verifier for {task}:\n{output}")
    return {"task": task, "training_provider_calls": len(log.calls)}


TASKS_DIR: Path = Path("")


async def run_llm_agent(model: str, temperature: float, max_steps: int, system_prompt: str,
                        openai_tools: list[dict[str, Any]], client: Client,
                        task_text: str, root: Path,
                        ledger_context: dict[str, Any] | None = None,
                        result_limit: int = 8000) -> tuple[CallLog, str]:
    from .gemini_rest import GeminiRestClient, provider_name
    from .token_usage import RequestLedger, as_token_usage
    provider = provider_name()
    use_gemini = provider == "gemini"
    llm = GeminiRestClient(model=model) if use_gemini else _openai_client()
    log = CallLog()
    ledger = RequestLedger()
    ctx = ledger_context or {}
    experiment_id = str(ctx.get("experiment_id", "mcpmark_file_property_ab"))
    ledger_task = str(ctx.get("task_id", ""))
    arm = str(ctx.get("arm", ""))
    attempt_no = int(ctx.get("attempt", 0))

    def _record_request(payload: Any, *, ok: bool = True, error: str = "",
                        is_retry: bool = False) -> None:
        usage = as_token_usage(payload, provider)
        record = ledger.log(
            experiment_id=experiment_id, task_id=ledger_task, arm=arm,
            attempt=attempt_no, model=model, provider=provider,
            usage=usage, ok=ok, is_retry=is_retry, error=error)
        log.request_rows.append(record.to_dict())
        log.raw_usage.append(usage.to_dict())
        if usage.is_complete:
            log.prompt_tokens += usage.input_tokens or 0
            log.completion_tokens += usage.output_tokens or 0

    if use_gemini:
        contents: list[dict[str, Any]] = [
            {"role": "user", "parts": [{"text": f"{task_text}\n\nWorkspace root: {root}"}]}]
        steps = 0
        while steps < max_steps:
            steps += 1
            try:
                response = await asyncio.to_thread(
                    llm.generate, system_prompt, contents, openai_tools, temperature, 1024)
            except Exception as exc:
                _record_request(None, ok=False, error=f"{type(exc).__name__}: {exc}",
                                is_retry=steps > 1)
                raise
            _record_request(response.get("usage") or response.get("raw_usage") or {
                "promptTokenCount": response.get("prompt_tokens"),
                "candidatesTokenCount": response.get("completion_tokens")},
                is_retry=False)
            if not response["tool_calls"]:
                return log, response["text"]
            # Echo raw parts verbatim: preserves thoughtSignature required by Gemini 3+.
            contents.append({"role": "model", "parts": response["raw_parts"]})
            response_parts = []
            for tc in response["tool_calls"]:
                try:
                    args = json.loads(tc["args"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                try:
                    result = await client.call_tool(tc["name"], args)
                    log.calls.append(tc["name"])
                    payload = _content(result) or str(_value(result))
                    if result.is_error:
                        log.failed_provider_calls.append(tc["name"])
                        payload = f"MCP tool error: {payload}"
                except Exception as exc:
                    log.calls.append(tc["name"])
                    log.failed_provider_calls.append(tc["name"])
                    payload = f"MCP tool error: {exc}"
                response_parts.append({"functionResponse": {
                    "name": tc["name"], "response": {"result": truncate_result(payload, result_limit)}}})
            contents.append({"role": "user", "parts": response_parts})
        return log, ""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{task_text}\n\nWorkspace root: {root}"}]
    steps = 0
    while steps < max_steps:
        steps += 1
        try:
            response = await asyncio.to_thread(
                llm.chat.completions.create, model=model, messages=messages,
                tools=openai_tools or None, tool_choice="auto" if openai_tools else "none",
                temperature=temperature, max_tokens=1024)
        except Exception as exc:
            _record_request(None, ok=False, error=f"{type(exc).__name__}: {exc}",
                            is_retry=steps > 1)
            raise
        usage = getattr(response, "usage", None)
        try:
            raw = usage.model_dump() if usage is not None and hasattr(usage, "model_dump") else usage
        except Exception:
            raw = None
        _record_request(raw)
        choice = response.choices[0].message
        if not choice.tool_calls:
            return log, choice.content or ""
        messages.append({"role": "assistant", "content": choice.content, "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in choice.tool_calls]})
        for tc in choice.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                result = await client.call_tool(tc.function.name, args)
                log.calls.append(tc.function.name)
                payload = _content(result) or str(_value(result))
                if result.is_error:
                    log.failed_provider_calls.append(tc.function.name)
                    payload = f"MCP tool error: {payload}"
            except Exception as exc:
                log.calls.append(tc.function.name)
                log.failed_provider_calls.append(tc.function.name)
                payload = f"MCP tool error: {exc}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": truncate_result(payload, result_limit)})
    return log, ""


async def eval_attempt(task: str, arm: str, attempt: int, model: str, temperature: float,
                        snapshot: Path, workdir: Path, frozen_db: Path, out_root: Path,
                        guidance_cap: int = 384,
                        result_limit: int = 8000) -> dict[str, Any]:
    from .filesystem_demo import filesystem_server_path as _fsp
    root = restore_snapshot(snapshot, out_root / f"{task}-{arm}-k{attempt}")
    node = shutil.which("node")
    fs_params = StdioServerParameters(
        command=node, args=[str(_fsp(Path.cwd()).resolve()), str(root)])
    memory_server = create_memory_server(frozen_db, read_only=True)
    memory_calls = 0
    injected_guidance: dict[str, Any] | None = None
    started = time.perf_counter()
    async with Client(fs_params) as client, Client(memory_server) as memory:
        listed = await client.list_tools()
        openai_tools = [{"type": "function", "function": {
            "name": t.name, "description": t.description or "",
            "parameters": t.input_schema or {"type": "object", "properties": {}}}} for t in listed.tools]
        prompt = SYSTEM_PROMPT
        guidance_text = ""
        guidance_tokens_estimate = 0
        schema_audit: dict[str, Any] | None = None
        schema_tokens_before = estimate_schema_tokens(openai_tools)
        schema_tokens_after = schema_tokens_before
        agent_tools = openai_tools
        if arm == "toolatlas":
            guidance = _value(await memory.call_tool(
                "get_guidance", {"task": TRAIN_SUMMARIES[task], "top_k": 3,
                                 "read_budget": 8, "token_budget": 384}))
            memory_calls += 1
            injected_guidance = guidance.get("playbook", [])
            guidance_text = render_compact_block(guidance, token_budget=guidance_cap)
            guidance_tokens_estimate = estimate_guidance_tokens(guidance_text)
            if guidance_text:
                prompt += f"\n{guidance_text}"
            agent_tools, schema_audit = filter_tools_by_playbook(openai_tools, guidance)
            schema_tokens_after = estimate_schema_tokens(agent_tools)
        task_text = (TASKS_DIR / task / "description.md").read_text()
        log, final = await run_llm_agent(
            model, temperature, 30, prompt, agent_tools, client, task_text, root,
            {"experiment_id": "mcpmark_file_property_ab", "task_id": task,
             "arm": arm, "attempt": attempt},
            result_limit=result_limit)
    passed, output = run_verifier(TASKS_DIR / task / "verify.py", root)
    authoritative = all((row.get("usage") or {}).get("has_authoritative_input")
                        for row in log.request_rows) if log.request_rows else False
    return {"task": task, "arm": arm, "attempt": attempt, "passed": passed,
            "provider_tool_calls": len(log.calls), "memory_tool_calls": memory_calls,
            "total_mcp_calls": len(log.calls) + memory_calls,
            "failed_provider_calls": len(log.failed_provider_calls),
            "tools_used": log.calls, "prompt_tokens": log.prompt_tokens,
            "completion_tokens": log.completion_tokens, "total_tokens": log.total_tokens,
            "raw_usage": log.raw_usage, "request_rows": log.request_rows,
            "model_requests": len(log.request_rows),
            "has_complete_input_usage": authoritative,
            "injected_guidance": injected_guidance,
            "guidance_text": guidance_text,
            "guidance_tokens_estimate": guidance_tokens_estimate,
            "schema_pruning_audit": schema_audit,
            "schema_tokens_before": schema_tokens_before,
            "schema_tokens_after": schema_tokens_after,
            "seconds": round(time.perf_counter() - started, 1),
            "final_text": final[:500], "verifier_output": output[-500:]}


async def preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Fail-fast validation: snapshot, tasks, training, frozen memory, LLM probe. No eval runs."""
    import tempfile
    checks: dict[str, Any] = {}
    snapshot = args.snapshot.resolve()
    tasks_dir = args.tasks_dir.resolve()
    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    checks["snapshot_sha256_ok"] = digest == SNAPSHOT_SHA256
    checks["snapshot_sha256"] = digest

    tasks = ["size_classification", "time_classification"]
    files_ok = True
    verify_hashes = {}
    for task in tasks:
        for name in ("description.md", "verify.py"):
            path = tasks_dir / task / name
            ok = path.is_file()
            files_ok = files_ok and ok
            if name == "verify.py" and ok:
                verify_hashes[task] = hashlib.sha256(path.read_bytes()).hexdigest()
    checks["task_files_ok"] = files_ok
    checks["verifier_sha256"] = verify_hashes

    # Restore preserves baked mtimes (time task depends on them).
    probe_root = restore_snapshot(snapshot, Path(tempfile.mkdtemp(prefix="fp-preflight")) / "root")
    sg = probe_root / "sg.jpg"
    checks["mtime_preserved"] = (
        sg.is_file()
        and datetime.fromtimestamp(sg.stat().st_mtime).strftime("%m/%d") == "07/09")

    # Full deterministic training (MCP only, no LLM) + frozen memory + guidance.
    global TASKS_DIR
    TASKS_DIR = tasks_dir
    work = Path(tempfile.mkdtemp(prefix="fp-preflight-train"))
    train_db = work / "train.db"
    memory_server = create_memory_server(train_db)
    async with Client(memory_server) as memory:
        node = shutil.which("node")
        from .filesystem_demo import filesystem_server_path as _fsp
        probe_params = StdioServerParameters(
            command=node, args=[str(_fsp(Path.cwd()).resolve()), str(probe_root)])
        async with Client(probe_params) as probe:
            tools = await probe.list_tools()
        await memory.call_tool("register_tools", {"tools": [
            {"name": t.name, "description": t.description or "", "input_schema": t.input_schema,
             "provider": PROVIDER, "version": VERSION} for t in tools.tools]})
        train_report = [await train_task(t, snapshot, work, memory) for t in tasks]
        playbooks = {}
        for t in tasks:
            guidance = _value(await memory.call_tool(
                "get_guidance", {"task": TRAIN_SUMMARIES[t], "top_k": 3, "read_budget": 8}))
            playbooks[t] = [s["tool"] for s in guidance.get("playbook", [])]
        checks["training"] = train_report
        checks["stats"] = _value(await memory.call_tool("memory_stats", {}))
    checks["playbooks"] = playbooks
    checks["playbooks_nonempty"] = all(playbooks.values())

    # One cheap LLM probe incl. tool-calling support (provider-aware).
    try:
        from .gemini_rest import GeminiRestClient, provider_name
        if provider_name() == "gemini":
            gemini = GeminiRestClient(model=args.model)
            probe = await asyncio.to_thread(
                gemini.generate, "Reply with the word ok.",
                [{"role": "user", "parts": [{"text": "Reply with the word ok."}]}],
                None, 0.0, 16)
            checks["llm_ok"] = True
            checks["llm_probe_text"] = probe["text"][:50]
            tool_probe = await asyncio.to_thread(
                gemini.generate, "Call the ping tool.",
                [{"role": "user", "parts": [{"text": "Call the ping tool."}]}],
                [{"function": {"name": "ping", "description": "test tool",
                               "parameters": {"type": "object", "properties": {}}}}],
                0.0, 64)
            checks["llm_tool_calling"] = any(tc["name"] == "ping" for tc in tool_probe["tool_calls"])
        else:
            llm = _openai_client()
            probe_response = await asyncio.to_thread(
                llm.chat.completions.create, model=args.model,
                messages=[{"role": "user", "content": "Reply with the word ok."}],
                temperature=0, max_tokens=16)
            checks["llm_ok"] = True
            checks["llm_probe_text"] = (probe_response.choices[0].message.content or "")[:50]
            tool_probe = await asyncio.to_thread(
                llm.chat.completions.create, model=args.model,
                messages=[{"role": "user", "content": "Call the ping tool."}],
                tools=[{"type": "function", "function": {
                    "name": "ping", "description": "test tool",
                    "parameters": {"type": "object", "properties": {}}}},
                ], tool_choice="auto", temperature=0, max_tokens=64)
            tool_calls = tool_probe.choices[0].message.tool_calls or []
            checks["llm_tool_calling"] = any(tc.function.name == "ping" for tc in tool_calls)
    except Exception as exc:
        checks["llm_ok"] = False
        checks["llm_tool_calling"] = False
        checks["llm_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"

    checks["preflight_passed"] = bool(
        checks["snapshot_sha256_ok"] and checks["task_files_ok"]
        and checks["mtime_preserved"] and checks["playbooks_nonempty"]
        and checks.get("llm_ok") and checks.get("llm_tool_calling"))
    print(json.dumps(checks, indent=2))
    return checks


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    global TASKS_DIR
    snapshot = args.snapshot.resolve()
    TASKS_DIR = args.tasks_dir.resolve()
    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    check_snapshot(snapshot)
    tasks = ["size_classification", "time_classification"]
    for task in tasks:
        if not (TASKS_DIR / task / "description.md").is_file() or not (TASKS_DIR / task / "verify.py").is_file():
            raise FileNotFoundError(f"official task files missing for {task} in {TASKS_DIR}")

    from .gemini_rest import provider_name
    try:
        code_rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True, timeout=15).stdout.strip()
    except Exception:
        code_rev = "unknown"
    manifest = {"model": args.model, "temperature": args.temperature, "k": args.k,
                "max_steps": 30, "provider": provider_name(),
                "guidance_cap": args.guidance_cap, "result_limit": args.result_limit,
                "snapshot_sha256": SNAPSHOT_SHA256, "code_rev": code_rev,
                "claim_scope": "official_snapshot_and_verifier_file_property_slice_frozen_memory_real_llm",
                "train_eval_overlap": "same-task deterministic training; label as same-task reuse, not held-out"}
    manifest_path = out_root / "manifest.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        mismatched = [key for key in ("model", "temperature", "k", "provider", "guidance_cap",
                                       "result_limit", "snapshot_sha256", "code_rev")
                      if prior.get(key) != manifest[key]]
        if mismatched:
            raise SystemExit(
                f"refusing to resume: manifest mismatch on {mismatched}. "
                "Use a fresh --output directory when changing model/settings/snapshot/code.")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))

    train_db = out_root / "train-memory.db"
    frozen_db = out_root / "frozen-memory.db"
    partial_log = out_root / "attempts_partial.jsonl"
    freeze_record: dict[str, Any] = {}
    if args.resume and frozen_db.is_file():
        print("resume: reusing frozen memory, skipping training", flush=True)
        memory_server = create_memory_server(frozen_db)
        async with Client(memory_server) as memory:
            stats = _value(await memory.call_tool("memory_stats", {}))
        train_report = [{"resumed": True}]
        digest = hashlib.sha256()
        with open(frozen_db, "rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        freeze_record = {"resumed": True, "frozen_sha256": digest.hexdigest()}
    else:
        if train_db.exists():
            train_db.unlink()
        memory_server = create_memory_server(train_db)
        async with Client(memory_server) as memory:
            listed_specs: list[dict[str, Any]] = []
            node = shutil.which("node")
            from .filesystem_demo import filesystem_server_path as _fsp
            probe_root = restore_snapshot(snapshot, out_root / "probe")
            probe_params = StdioServerParameters(
                command=node, args=[str(_fsp(Path.cwd()).resolve()), str(probe_root)])
            async with Client(probe_params) as probe:
                for tool in (await probe.list_tools()).tools:
                    listed_specs.append({"name": tool.name, "description": tool.description or "",
                                         "input_schema": tool.input_schema, "provider": PROVIDER, "version": VERSION})
            await memory.call_tool("register_tools", {"tools": listed_specs})
            train_report = []
            for task in tasks:
                train_report.append(await train_task(task, snapshot, out_root, memory))
            guidance_check = _value(await memory.call_tool(
                "get_guidance", {"task": TRAIN_SUMMARIES["size_classification"], "top_k": 3, "read_budget": 8}))
            stats = _value(await memory.call_tool("memory_stats", {}))
        # Freeze via checkpoint + backup API (never a raw file copy: WAL
        # content not yet checkpointed would silently vanish from the copy).
        from .freeze_memory import freeze_memory
        freeze_record = freeze_memory(train_db, frozen_db)
        if partial_log.is_file():
            partial_log.unlink()

    attempts: list[dict[str, Any]] = []
    done_keys: set[tuple[str, str, int]] = set()
    if args.resume and partial_log.is_file():
        with open(partial_log) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    attempts.append(record)
                    done_keys.add((record["task"], record["arm"], record["attempt"]))
        print(f"resume: loaded {len(attempts)} completed attempts", flush=True)

    with open(partial_log, "a") as fh:
        for task in tasks:
            for arm in ("baseline", "toolatlas"):
                for attempt in range(1, args.k + 1):
                    if (task, arm, attempt) in done_keys:
                        print(f"[{task} {arm} k={attempt}/{args.k}] skipping (done)", flush=True)
                        continue
                    print(f"[{task} {arm} k={attempt}/{args.k}] running...", flush=True)
                    record = await eval_attempt(
                        task, arm, attempt, args.model, args.temperature,
                        snapshot, out_root, frozen_db, out_root,
                        guidance_cap=args.guidance_cap,
                        result_limit=args.result_limit)
                    attempts.append(record)
                    fh.write(json.dumps(record) + "\n")
                    fh.flush()

    def summarize(task: str, arm: str) -> dict[str, Any]:
        rows = [a for a in attempts if a["task"] == task and a["arm"] == arm]
        passed = sum(1 for a in rows if a["passed"])
        provider = [a["provider_tool_calls"] for a in rows]
        total = [a["total_mcp_calls"] for a in rows]
        tokens = [a["total_tokens"] for a in rows]
        return {"task": task, "arm": arm, "n": len(rows), "passed": passed,
                "pass_rate": round(passed / len(rows), 4),
                "avg_provider_calls": round(sum(provider) / len(provider), 2),
                "avg_total_calls": round(sum(total) / len(total), 2),
                "avg_tokens": round(sum(tokens) / len(tokens), 1) if any(tokens) else 0}

    summary = [summarize(t, a) for t in tasks for a in ("baseline", "toolatlas")]
    from .freeze_memory import verify_frozen_unchanged
    frozen_check = verify_frozen_unchanged(
        frozen_db, freeze_record.get("frozen_sha256", ""))
    report = {"benchmark": "mcpmark_file_property_llm_ab", "model": args.model,
              "temperature": args.temperature, "k": args.k,
              "manifest": manifest,
              "snapshot_sha256": SNAPSHOT_SHA256, "frozen_memory_stats": stats,
              "training": train_report, "freeze": freeze_record,
              "frozen_post_eval": frozen_check,
              "attempts": attempts, "summary": summary}
    (out_root / "report.json").write_text(json.dumps(report, indent=2))
    lines = ["# MCPMark file_property LLM A/B", "",
             f"Model: {args.model}, temperature {args.temperature}, k={args.k}", "",
             "| Task | Arm | Passed | Avg provider calls | Avg total calls | Avg tokens |",
             "|---|---|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['task']} | {row['arm']} | {row['passed']}/{row['n']} "
                     f"| {row['avg_provider_calls']} | {row['avg_total_calls']} | {row['avg_tokens']} |")
    (out_root / "report.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-LLM both-arms A/B on MCPMark file_property (official snapshot+verifier)")
    parser.add_argument("--snapshot", type=Path, default=Path("benchmarks/official/file_property.zip"))
    parser.add_argument("--tasks-dir", type=Path, default=Path("benchmarks/official/file_property"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL") or os.environ.get("NVIDIA_MODEL", "deepseek-chat"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--guidance-cap", type=int, default=384,
                        help="post-render cap for the compact injected guidance block (Plan 1 tuning: 128/256)")
    parser.add_argument("--result-limit", type=int, default=8000,
                        help="shared deterministic tool-result char limit, both arms (Plan 2 ablation)")
    parser.add_argument("--resume", action="store_true",
                        help="reuse frozen memory and skip attempts already in attempts_partial.jsonl")
    parser.add_argument("--preflight", action="store_true",
                        help="validate snapshot/tasks/training/frozen-memory/LLM without running eval attempts")
    args = parser.parse_args()
    if args.preflight:
        report = asyncio.run(preflight(args))
        if not report.get("preflight_passed"):
            raise SystemExit(1)
        return
    report = asyncio.run(main_async(args))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
