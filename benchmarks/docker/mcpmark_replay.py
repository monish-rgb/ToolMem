"""Deterministic MCP replay of an official task, NOT an LLM benchmark score.

Only snapshot setup and the independent official verifier access files directly.
The task policy reads sizes and performs all mutations through MCP stdio.
"""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

from mcp import Client, StdioServerParameters
from toolatlas.filesystem_demo import filesystem_server_path

SNAPSHOT = Path('/opt/snapshots/file_property.zip')
VERIFIER = Path('/app/tasks/filesystem/file_property/size_classification/verify.py')


def content(result):
    return '\n'.join(getattr(item, 'text', '') for item in result.content)


async def main():
    attempts = []
    for attempt in range(4):
        root = Path(f'mcp-sandbox/mcpmark-size-{attempt}').resolve()
        root.mkdir(parents=True)
        with zipfile.ZipFile(SNAPSHOT) as archive:
            for item in archive.infolist():
                parts = Path(item.filename).parts
                if len(parts) != 2 or parts[0] != 'file_property' or item.is_dir():
                    continue
                target = root / parts[1]
                assert target.resolve().is_relative_to(root)
                target.write_bytes(archive.read(item))
        calls = []
        start = time.monotonic()
        params = StdioServerParameters(command='node', args=[str(filesystem_server_path(Path.cwd())), str(root)])
        async with Client(params) as client:
            async def call(name, args):
                result = await client.call_tool(name, args)
                assert not result.is_error, name
                calls.append(name)
                return content(result)
            listing = await call('list_directory', {'path': str(root)})
            files = [line.removeprefix('[FILE] ') for line in listing.splitlines() if line.startswith('[FILE] ')]
            assert files
            for category in ['small_files', 'medium_files', 'large_files']:
                await call('create_directory', {'path': str(root / category)})
            for name in files:
                info = await call('get_file_info', {'path': str(root / name)})
                match = re.search(r'^size:\s*(\d+)', info, re.MULTILINE | re.IGNORECASE)
                assert match, info
                size = int(match[1])
                category = 'small_files' if size < 300 else 'medium_files' if size <= 700 else 'large_files'
                await call('move_file', {'source': str(root / name), 'destination': str(root / category / name)})
        verified = subprocess.run([sys.executable, str(VERIFIER)], env={**os.environ, 'FILESYSTEM_TEST_DIR': str(root)},
                                  capture_output=True, text=True, timeout=30)
        attempts.append(dict(attempt=attempt + 1, passed=verified.returncode == 0,
            provider_tool_calls=len(calls), memory_tool_calls=0, total_mcp_calls=len(calls),
            tool_calls=calls, seconds=round(time.monotonic()-start, 3), verifier_output=verified.stdout))
    report = dict(task='filesystem/file_property/size_classification',
        claim_scope='official_task_snapshot_and_verifier_with_custom_deterministic_policy_not_llm_benchmark',
        is_toolatlas_assisted=False, independent_llm_rollouts=False, inference_tokens=0,
        snapshot_sha256=hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(),
        verifier_sha256=hashlib.sha256(VERIFIER.read_bytes()).hexdigest(), attempts=attempts)
    print(json.dumps(report, indent=2))
    assert all(a['passed'] for a in attempts)


asyncio.run(main())
