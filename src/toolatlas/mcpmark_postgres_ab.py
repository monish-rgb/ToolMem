"""Real-LLM both-arms A/B over MCPMark's Chinook employee-hierarchy task.

Pipeline (mirrors mcpmark_file_property_ab.py):
  1. TRAIN (deterministic, verified): run a scripted SQL policy through the
     postgres MCP server on a training database, check the official
     verifier, induce two same-sequence traces into ToolAtlas memory so a
     strategy forms. Then FREEZE the DB.
  2. EVAL (real LLM, k independent rollouts per arm): fresh database per
     attempt (drop + create + pg_restore of the pinned backup). Arm A =
     postgres tools only. Arm B = one get_guidance call against the frozen
     DB, then the same tools. No ingestion during eval. Each attempt is
     checked by the official verify.py against the attempt database.

Requires: Docker container running PostgreSQL 17 with the backup staged at
/tmp/chinook.backup (see setup below), postgres-mcp binary (pinned to
mcp<2 in its own pipx venv), and an LLM provider (LLM_*/GEMINI_*/NVIDIA_*).

Setup (Windows cmd):
  docker run -d --name mcpmark-postgres -e POSTGRES_PASSWORD=mysecretpassword ^
    -e POSTGRES_USER=postgres -p 5432:5432 postgres:17
  docker cp benchmarks/official/postgres/chinook.backup mcpmark-postgres:/tmp/chinook.backup
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from .llm_ab_nim import _openai_client
from .memory_server import create_memory_server
from .readonly_benchmark import _value
from .guidance_render import estimate_tokens as estimate_guidance_tokens
from .guidance_render import render_compact_block
from .result_limit import truncate_result
from .tool_filter import estimate_schema_tokens, filter_tools_by_playbook

BACKUP_SHA256 = "50C7969A9E8F2CC1CE250AF58FC46D0ABEFC196C10A006DE979008443F324A14"
PROVIDER = "crystaldba/postgres-mcp"
VERSION = "pinned-mcp-v1"
CONTAINER = "mcpmark-postgres"
PG_USER = "postgres"
PG_PASSWORD = "mysecretpassword"
PG_HOST = "127.0.0.1"
PG_PORT = 5432
MCP_BIN = r"C:\Users\MONISH\.local\bin\postgres-mcp.EXE"

SYSTEM_PROMPT = (
    "You are a PostgreSQL agent with MCP tools. Complete the user's task "
    "against the connected database using only the provided tools. When finished, "
    "reply with a short summary of what you did."
)
TRAIN_SUMMARY = "Modify employee hierarchy and customer assignments with SQL CRUD operations"

TRAIN_STEPS = [
    {"tool": "get_object_details", "rationale": "inspect table schemas before writing SQL"},
    {"tool": "execute_sql", "rationale": "apply verified data changes with SQL statements"},
]


@dataclass
class CallLog:
    calls: list[str] = field(default_factory=list)
    invalid_tool_calls: list[str] = field(default_factory=list)
    failed_provider_calls: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_usage: list[dict[str, Any]] = field(default_factory=list)
    request_rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=300)


def check_backup(backup: Path) -> None:
    digest = hashlib.sha256(backup.read_bytes()).hexdigest()
    if digest.lower() != BACKUP_SHA256.lower():
        raise AssertionError(f"backup SHA-256 mismatch: {digest}")


def reset_db(dbname: str) -> None:
    """Drop + recreate + restore. Full restore per runbook (no bare rollback)."""
    docker("exec", "-e", f"PGPASSWORD={PG_PASSWORD}", CONTAINER,
           "dropdb", "-U", PG_USER, "--if-exists", dbname).check_returncode()
    docker("exec", "-e", f"PGPASSWORD={PG_PASSWORD}", CONTAINER,
           "createdb", "-U", PG_USER, dbname).check_returncode()
    result = docker("exec", "-e", f"PGPASSWORD={PG_PASSWORD}", CONTAINER,
                    "pg_restore", "-U", PG_USER, "-d", dbname, "/tmp/chinook.backup")
    if result.returncode != 0:
        raise AssertionError(f"pg_restore failed for {dbname}: {result.stderr[-500:]}")


def db_uri(dbname: str) -> str:
    return f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{dbname}"


def mcp_params(dbname: str) -> StdioServerParameters:
    binary = os.environ.get("POSTGRES_MCP_BIN", MCP_BIN)
    return StdioServerParameters(
        command=binary, args=["--access-mode=unrestricted"],
        env={"PATH": os.environ.get("PATH", ""), "DATABASE_URI": db_uri(dbname)})


def run_verifier(verify_py: Path, dbname: str) -> tuple[bool, str]:
    env = {**os.environ, "POSTGRES_HOST": PG_HOST, "POSTGRES_PORT": str(PG_PORT),
           "POSTGRES_DATABASE": dbname, "POSTGRES_USERNAME": PG_USER,
           "POSTGRES_PASSWORD": PG_PASSWORD, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run([sys.executable, str(verify_py)], env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=120)
    return proc.returncode == 0, ((proc.stdout or "") + (proc.stderr or ""))[-2000:]


def _sql(result: Any) -> str:
    blocks = getattr(result, "content", None) or []
    return "\n".join(str(getattr(block, "text", "")) for block in blocks)


POLICY_SQL = [
    "INSERT INTO \"Employee\" (\"EmployeeId\", \"FirstName\", \"LastName\", \"Title\", \"ReportsTo\", "
    "\"BirthDate\", \"HireDate\", \"Address\", \"City\", \"State\", \"Country\", \"PostalCode\", \"Phone\", "
    "\"Fax\", \"Email\") VALUES (9, 'Sarah', 'Johnson', 'Sales Support Agent', 2, '1985-03-15', '2009-01-10', "
    "'123 Oak Street', 'Calgary', 'AB', 'Canada', 'T2P 5G3', '+1 (403) 555-0123', '+1 (403) 555-0124', "
    "'sarah.johnson@chinookcorp.com'), (10, 'Mike', 'Chen', 'Sales Support Agent', 2, '1982-08-22', '2009-01-10', "
    "'456 Pine Ave', 'Calgary', 'AB', 'Canada', 'T2P 5G4', '+1 (403) 555-0125', '+1 (403) 555-0126', "
    "'mike.chen@chinookcorp.com');",
    "UPDATE \"Employee\" SET \"Title\" = 'CEO' WHERE \"EmployeeId\" = 1;",
    "UPDATE \"Employee\" SET \"Phone\" = '+1 (403) 555-9999' WHERE \"EmployeeId\" = 2;",
    "UPDATE \"Employee\" SET \"Title\" = 'IT Specialist' WHERE \"Title\" = 'IT Staff';",
    "UPDATE \"Customer\" SET \"SupportRepId\" = 9 WHERE \"CustomerId\" IN (1, 2, 3);",
    "UPDATE \"Customer\" SET \"SupportRepId\" = 10 WHERE \"CustomerId\" IN (4, 5, 6);",
    "UPDATE \"Employee\" SET \"ReportsTo\" = 1 WHERE \"EmployeeId\" IN (9, 10);",
    "CREATE TABLE employee_performance (employee_id integer REFERENCES \"Employee\"(\"EmployeeId\"), "
    "customers_assigned integer, performance_score decimal);",
    "INSERT INTO employee_performance (employee_id, customers_assigned, performance_score) "
    "SELECT 9, COUNT(*), 4.5 FROM \"Customer\" WHERE \"SupportRepId\" = 9;",
    "INSERT INTO employee_performance (employee_id, customers_assigned, performance_score) "
    "SELECT 10, COUNT(*), 4.2 FROM \"Customer\" WHERE \"SupportRepId\" = 10;",
    "UPDATE \"Employee\" SET \"ReportsTo\" = (SELECT \"ReportsTo\" FROM \"Employee\" WHERE \"EmployeeId\" = 7) "
    "WHERE \"ReportsTo\" = 7;",
    "UPDATE \"Customer\" SET \"SupportRepId\" = (SELECT \"ReportsTo\" FROM \"Employee\" WHERE \"EmployeeId\" = 7) "
    "WHERE \"SupportRepId\" = 7;",
    "DELETE FROM \"Employee\" WHERE \"EmployeeId\" = 7;",
    "ALTER TABLE \"Employee\" ADD COLUMN salary decimal;",
    "UPDATE \"Employee\" SET salary = 75000.00 WHERE \"EmployeeId\" = 8;",
    "UPDATE \"Employee\" SET salary = 50000.00 WHERE \"EmployeeId\" <> 8;",
    "UPDATE \"Employee\" SET \"Title\" = 'Senior IT Specialist' WHERE \"EmployeeId\" = 8;",
]


async def train_task(task_dir: Path, memory: Client, log: CallLog) -> dict[str, Any]:
    reset_db("chinook_train")
    async with Client(mcp_params("chinook_train")) as client:
        details = await client.call_tool("get_object_details", {"schema_name": "public", "object_name": "Employee"})
        if details.is_error:
            raise AssertionError("training policy failed at get_object_details")
        log.calls.append("get_object_details")
        for sql in POLICY_SQL:
            result = await client.call_tool("execute_sql", {"sql": sql})
            if result.is_error:
                raise AssertionError(f"training policy failed at: {sql[:80]} -> {_sql(result)[:200]}")
            log.calls.append("execute_sql")
    passed, output = run_verifier(task_dir / "verify.py", "chinook_train")
    for run in (1, 2):
        await memory.call_tool("remember_execution", {
            "task_id": f"mcpmark_employee_hierarchy_{run}", "summary": TRAIN_SUMMARY,
            "steps": TRAIN_STEPS, "resolved": passed,
            "observation": "the official verifier accepted the reorganized hierarchy",
            "verifier_type": "official_mcpmark_verifier"})
    if not passed:
        raise AssertionError(f"training policy failed verifier:\n{output}")
    return {"training_provider_calls": len(log.calls)}


async def run_llm_agent(model: str, temperature: float, max_steps: int, system_prompt: str,
                        openai_tools: list[dict[str, Any]], client: Client,
                        task_text: str,
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
    experiment_id = str(ctx.get("experiment_id", "mcpmark_postgres_chinook_ab"))
    ledger_task = str(ctx.get("task_id", "employee_hierarchy_management"))
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

    allowed_tool_names = {
        (tool.get("function") or {}).get("name")
        for tool in openai_tools
        if isinstance(tool, dict)
    }
    allowed_tool_names.discard(None)
    if use_gemini:
        contents: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": task_text}]}]
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
                "candidatesTokenCount": response.get("completion_tokens")})
            if not response["tool_calls"]:
                return log, response["text"]
            contents.append({"role": "model", "parts": response["raw_parts"]})
            response_parts = []
            for tc in response["tool_calls"]:
                try:
                    args = json.loads(tc["args"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                try:
                    if tc["name"] not in allowed_tool_names:
                        log.invalid_tool_calls.append(tc["name"])
                        payload = (
                            f"MCP tool error: unknown tool {tc['name']!r}. "
                            "Use only the declared PostgreSQL tool names."
                        )
                    else:
                        result = await client.call_tool(tc["name"], args)
                        log.calls.append(tc["name"])
                        payload = _sql(result) or str(_value(result))
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
        {"role": "user", "content": task_text}]
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
                if tc.function.name not in allowed_tool_names:
                    log.invalid_tool_calls.append(tc.function.name)
                    payload = (
                        f"MCP tool error: unknown tool {tc.function.name!r}. "
                        "Use only the declared PostgreSQL tool names."
                    )
                else:
                    result = await client.call_tool(tc.function.name, args)
                    log.calls.append(tc.function.name)
                    payload = _sql(result) or str(_value(result))
                    if result.is_error:
                        log.failed_provider_calls.append(tc.function.name)
                        payload = f"MCP tool error: {payload}"
            except Exception as exc:
                log.calls.append(tc.function.name)
                log.failed_provider_calls.append(tc.function.name)
                payload = f"MCP tool error: {exc}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": truncate_result(payload, result_limit)})
    return log, ""


def _exception_summary(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        parts = [_exception_summary(inner) for inner in exc.exceptions]
        return "; ".join(part for part in parts if part)[:1000]
    return f"{type(exc).__name__}: {exc}"[:1000]


async def eval_attempt(arm: str, attempt: int, model: str, temperature: float,
                        task_dir: Path, frozen_db: Path, out_root: Path,
                        guidance_cap: int = 384,
                        result_limit: int = 8000) -> dict[str, Any]:
    dbname = f"chinook_{arm}_k{attempt}"
    reset_db(dbname)
    # Evaluation opens frozen memory read-only: even SQLite housekeeping
    # writes (metadata upsert on open) would change the file hash and trip
    # the post-eval immutability audit.
    memory_server = create_memory_server(frozen_db, read_only=True)
    memory_calls = 0
    injected_guidance: dict[str, Any] | None = None
    guidance_text = ""
    guidance_tokens_estimate = 0
    started = time.perf_counter()
    log = CallLog()
    final = ""
    error: str | None = None
    try:
        async with Client(mcp_params(dbname)) as client, Client(memory_server) as memory:
            listed = await client.list_tools()
            openai_tools = [{"type": "function", "function": {
                "name": t.name, "description": t.description or "",
                "parameters": t.input_schema or {"type": "object", "properties": {}}}} for t in listed.tools]
            prompt = SYSTEM_PROMPT + (
                "\nUse only the declared function tools. Do not call raw MCP protocol methods "
                "such as server/discover, initialize, tools/list, or tools/call."
            )
            guidance_text = ""
            guidance_tokens_estimate = 0
            agent_tools = openai_tools
            if arm == "toolatlas":
                guidance = _value(await memory.call_tool(
                    "get_guidance", {"task": TRAIN_SUMMARY, "top_k": 3,
                                     "read_budget": 8, "token_budget": 384}))
                memory_calls += 1
                injected_guidance = guidance.get("playbook", [])
                guidance_text = render_compact_block(guidance, token_budget=guidance_cap)
                guidance_tokens_estimate = estimate_guidance_tokens(guidance_text)
                if guidance_text:
                    prompt += f"\n{guidance_text}"
                agent_tools, _ = filter_tools_by_playbook(openai_tools, guidance)
            task_text = (task_dir / "description.md").read_text()
            log, final = await run_llm_agent(
                model, temperature, 30, prompt, agent_tools, client, task_text,
                {"experiment_id": "mcpmark_postgres_chinook_ab",
                 "task_id": "employee_hierarchy_management",
                 "arm": arm, "attempt": attempt},
                result_limit=result_limit)
    except Exception as exc:
        error = _exception_summary(exc)
    verifier_passed, output = run_verifier(task_dir / "verify.py", dbname)
    passed = verifier_passed and error is None
    authoritative = all((row.get("usage") or {}).get("has_authoritative_input")
                        for row in log.request_rows) if log.request_rows else False
    return {"arm": arm, "attempt": attempt, "passed": passed,
            "runtime_error": error, "verifier_passed": verifier_passed,
            "provider_tool_calls": len(log.calls), "memory_tool_calls": memory_calls,
            "total_mcp_calls": len(log.calls) + memory_calls,
            "failed_provider_calls": len(log.failed_provider_calls),
            "tools_used": log.calls, "invalid_tool_calls": log.invalid_tool_calls,
            "prompt_tokens": log.prompt_tokens,
            "completion_tokens": log.completion_tokens, "total_tokens": log.total_tokens,
            "raw_usage": log.raw_usage, "request_rows": log.request_rows,
            "model_requests": len(log.request_rows),
            "has_complete_input_usage": authoritative,
            "injected_guidance": injected_guidance,
            "guidance_text": guidance_text,
            "guidance_tokens_estimate": guidance_tokens_estimate,
            "seconds": round(time.perf_counter() - started, 1),
            "final_text": final[:500], "verifier_output": output[-500:]}


async def preflight(args: argparse.Namespace) -> dict[str, Any]:
    import tempfile
    checks: dict[str, Any] = {}
    backup = Path(args.backup).resolve()
    task_dir = Path(args.tasks_dir).resolve()
    digest = hashlib.sha256(backup.read_bytes()).hexdigest()
    checks["backup_sha256_ok"] = digest.lower() == BACKUP_SHA256.lower()
    checks["task_files_ok"] = all(
        (task_dir / name).is_file() for name in ("description.md", "verify.py", "meta.json"))
    checks["container_ok"] = docker("exec", CONTAINER, "pg_isready", "-U", PG_USER).returncode == 0
    try:
        reset_db("chinook_preflight")
        checks["restore_ok"] = True
    except AssertionError as exc:
        checks["restore_ok"] = False
        checks["restore_error"] = str(exc)[:300]
    out_tmp = Path(tempfile.mkdtemp(prefix="pg-preflight"))
    train_db = out_tmp / "train.db"
    memory_server = create_memory_server(train_db)
    async with Client(memory_server) as memory:
        async with Client(mcp_params("chinook_preflight")) as probe:
            tools = await probe.list_tools()
        checks["discovered_tools"] = sorted(t.name for t in tools.tools)
        await memory.call_tool("register_tools", {"tools": [
            {"name": t.name, "description": t.description or "", "input_schema": t.input_schema,
             "provider": PROVIDER, "version": VERSION} for t in tools.tools]})
        log = CallLog()
        report = await train_task(task_dir, memory, log)
        checks["training"] = report
        guidance = _value(await memory.call_tool(
            "get_guidance", {"task": TRAIN_SUMMARY, "top_k": 3, "read_budget": 8}))
        checks["playbook"] = [s["tool"] for s in guidance.get("playbook", [])]
        checks["playbook_nonempty"] = bool(checks["playbook"])
        checks["stats"] = _value(await memory.call_tool("memory_stats", {}))
    try:
        from .gemini_rest import GeminiRestClient, provider_name
        if provider_name() == "gemini":
            gemini = GeminiRestClient(model=args.model)
            probe = await asyncio.to_thread(
                gemini.generate, "Reply ok.", [{"role": "user", "parts": [{"text": "Reply ok."}]}],
                None, 0.0, 16)
            checks["llm_ok"] = True
            tool_probe = await asyncio.to_thread(
                gemini.generate, "Call ping.", [{"role": "user", "parts": [{"text": "Call ping."}]}],
                [{"function": {"name": "ping", "description": "t",
                               "parameters": {"type": "object", "properties": {}}}}], 0.0, 64)
            checks["llm_tool_calling"] = any(tc["name"] == "ping" for tc in tool_probe["tool_calls"])
        else:
            llm = _openai_client()
            probe_response = await asyncio.to_thread(
                llm.chat.completions.create, model=args.model,
                messages=[{"role": "user", "content": "Reply ok."}], temperature=0, max_tokens=16)
            checks["llm_ok"] = True
            tool_probe = await asyncio.to_thread(
                llm.chat.completions.create, model=args.model,
                messages=[{"role": "user", "content": "Call ping."}],
                tools=[{"type": "function", "function": {"name": "ping", "description": "t",
                        "parameters": {"type": "object", "properties": {}}}}],
                tool_choice="auto", temperature=0, max_tokens=64)
            checks["llm_tool_calling"] = any(
                tc.function.name == "ping" for tc in (tool_probe.choices[0].message.tool_calls or []))
    except Exception as exc:
        checks["llm_ok"] = False
        checks["llm_tool_calling"] = False
        checks["llm_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    checks["preflight_passed"] = bool(
        checks["backup_sha256_ok"] and checks["task_files_ok"] and checks.get("container_ok")
        and checks.get("restore_ok") and checks.get("playbook_nonempty")
        and checks.get("llm_ok") and checks.get("llm_tool_calling"))
    print(json.dumps(checks, indent=2))
    return checks


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    backup = Path(args.backup).resolve()
    task_dir = Path(args.tasks_dir).resolve()
    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    check_backup(backup)
    if not all((task_dir / name).is_file() for name in ("description.md", "verify.py")):
        raise FileNotFoundError(f"official task files missing in {task_dir}")

    from .gemini_rest import provider_name
    try:
        code_rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True, timeout=15).stdout.strip()
    except Exception:
        code_rev = "unknown"
    manifest = {"model": args.model, "temperature": args.temperature, "k": args.k,
                "max_steps": 30, "provider": provider_name(), "container": CONTAINER,
                "guidance_cap": args.guidance_cap, "result_limit": args.result_limit,
                "backup_sha256": BACKUP_SHA256, "code_rev": code_rev,
                "claim_scope": "official_backup_and_verifier_chinook_hierarchy_frozen_memory_real_llm",
                "train_eval_overlap": "same-task deterministic training; label as same-task reuse, not held-out"}
    manifest_path = out_root / "manifest.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        mismatched = [key for key in ("model", "temperature", "k", "provider", "guidance_cap",
                                       "result_limit", "backup_sha256", "code_rev")
                      if prior.get(key) != manifest[key]]
        if mismatched:
            raise SystemExit(f"refusing to resume: manifest mismatch on {mismatched}. Use a fresh --output.")
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
        train_report: Any = [{"resumed": True}]
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
            async with Client(mcp_params("chinook_train")) as probe:
                tools = await probe.list_tools()
            await memory.call_tool("register_tools", {"tools": [
                {"name": t.name, "description": t.description or "", "input_schema": t.input_schema,
                 "provider": PROVIDER, "version": VERSION} for t in tools.tools]})
            log = CallLog()
            train_report = [await train_task(task_dir, memory, log)]
            stats = _value(await memory.call_tool("memory_stats", {}))
        # Freeze via checkpoint + backup API (never a raw file copy: WAL
        # content not yet checkpointed would silently vanish from the copy).
        from .freeze_memory import freeze_memory
        freeze_record = freeze_memory(train_db, frozen_db)
        if partial_log.is_file():
            partial_log.unlink()

    attempts: list[dict[str, Any]] = []
    done_keys: set[tuple[str, int]] = set()
    if args.resume and partial_log.is_file():
        with open(partial_log) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    attempts.append(record)
                    done_keys.add((record["arm"], record["attempt"]))
        print(f"resume: loaded {len(attempts)} completed attempts", flush=True)

    with open(partial_log, "a") as fh:
        for arm in ("baseline", "toolatlas"):
            for attempt in range(1, args.k + 1):
                if (arm, attempt) in done_keys:
                    print(f"[{arm} k={attempt}/{args.k}] skipping (done)", flush=True)
                    continue
                print(f"[{arm} k={attempt}/{args.k}] running...", flush=True)
                record = await eval_attempt(arm, attempt, args.model, args.temperature,
                                            task_dir, frozen_db, out_root,
                                            guidance_cap=args.guidance_cap,
                                            result_limit=args.result_limit)
                attempts.append(record)
                fh.write(json.dumps(record) + "\n")
                fh.flush()

    def summarize(arm: str) -> dict[str, Any]:
        rows = [a for a in attempts if a["arm"] == arm]
        passed = sum(1 for a in rows if a["passed"])
        provider = [a["provider_tool_calls"] for a in rows]
        total = [a["total_mcp_calls"] for a in rows]
        tokens = [a["total_tokens"] for a in rows]
        return {"arm": arm, "n": len(rows), "passed": passed,
                "pass_rate": round(passed / len(rows), 4) if rows else 0,
                "avg_provider_calls": round(sum(provider) / len(provider), 2) if provider else 0,
                "avg_total_calls": round(sum(total) / len(total), 2) if total else 0,
                "avg_tokens": round(sum(tokens) / len(tokens), 1) if any(tokens) else 0}

    summary = [summarize(a) for a in ("baseline", "toolatlas")]
    from .freeze_memory import verify_frozen_unchanged
    frozen_check = verify_frozen_unchanged(
        frozen_db, freeze_record.get("frozen_sha256", ""))
    report = {"benchmark": "mcpmark_postgres_chinook_ab", "model": args.model,
              "temperature": args.temperature, "k": args.k, "manifest": manifest,
              "frozen_memory_stats": stats, "training": train_report,
              "freeze": freeze_record, "frozen_post_eval": frozen_check,
              "attempts": attempts, "summary": summary}
    (out_root / "report.json").write_text(json.dumps(report, indent=2))
    lines = ["# MCPMark Chinook hierarchy LLM A/B", "",
             f"Model: {args.model}, temperature {args.temperature}, k={args.k}", "",
             "| Arm | Passed | Avg provider calls | Avg total calls | Avg tokens |",
             "|---|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['arm']} | {row['passed']}/{row['n']} "
                     f"| {row['avg_provider_calls']} | {row['avg_total_calls']} | {row['avg_tokens']} |")
    (out_root / "report.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-LLM both-arms A/B on MCPMark Chinook hierarchy (official backup+verifier)")
    parser.add_argument("--backup", type=Path, default=Path("benchmarks/official/postgres/chinook.backup"))
    parser.add_argument("--tasks-dir", type=Path, default=Path("benchmarks/official/postgres/chinook/employee_hierarchy_management"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL") or os.environ.get("NVIDIA_MODEL", "deepseek-chat"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--guidance-cap", type=int, default=384,
                        help="post-render cap for the compact injected guidance block (Plan 1 tuning: 128/256)")
    parser.add_argument("--result-limit", type=int, default=8000,
                        help="shared deterministic tool-result char limit, both arms (Plan 2 ablation)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true",
                        help="validate container/restore/training/frozen-memory/LLM without eval attempts")
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
