"""Run local MCP checks in a disposable, network-disabled container."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

root = Path('/work/project')
root.mkdir()
for name in ('src', 'tests'):
    shutil.copytree(Path('/opt/toolatlas') / name, root / name)
for name in ('pyproject.toml', 'README.md', 'package.json'):
    shutil.copy(Path('/opt/toolatlas') / name, root / name)
(root / 'node_modules').symlink_to('/opt/toolatlas/node_modules')
(root / 'mcp-sandbox').mkdir()
os.chdir(root)
out = Path('/work/results')
out.mkdir()
commands = {
    'mcpmark-size-replay': ['/opt/toolatlas/benchmarks/docker/mcpmark_replay.py'],
    'tool-coverage': ['/opt/toolatlas/benchmarks/docker/tool_coverage.py'],
    'pytest': ['-m', 'pytest', '-q', '-ra', '--junitxml=/work/results/pytest.xml'],
    'compile': ['-m', 'compileall', '-q', 'src', 'tests'],
    'text-demo': ['-m', 'toolatlas.demo'],
    'filesystem-demo': ['-m', 'toolatlas.filesystem_demo'],
    'everything-demo': ['-m', 'toolatlas.everything_demo', '--root', str(root / 'mcp-sandbox')],
    'readonly-ab': ['-m', 'toolatlas.readonly_benchmark'],
    'paper-control': ['-m', 'toolatlas.paper_benchmark', '--output', str(out / 'paper-protocol-filesystem.json')],
}
results = []
for name, args in commands.items():
    start = time.monotonic()
    with (out / f'{name}.stdout').open('w') as stdout, (out / f'{name}.stderr').open('w') as stderr:
        try:
            result = subprocess.run([sys.executable, *args], stdout=stdout, stderr=stderr, timeout=600)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
    record = dict(check=name, exit_code=code, seconds=round(time.monotonic()-start, 3))
    results.append(record)
    print(json.dumps(record), flush=True)
(out / 'checks.json').write_text(json.dumps(results, indent=2) + '\n')
sys.exit(any(item['exit_code'] for item in results))
