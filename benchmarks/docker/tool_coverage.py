"""Verified stdio smoke coverage of every local memory and text tool."""
import asyncio
import base64
import json
from pathlib import Path
import sys
import time

from mcp import Client, StdioServerParameters
from toolatlas.demo import _value
from toolatlas.filesystem_demo import filesystem_server_path


async def main():
    records = []
    params = StdioServerParameters(command=sys.executable, args=['-m', 'toolatlas.memory_server'],
        env={'TOOLATLAS_MEMORY_PATH': str(Path('.toolatlas/coverage.db').resolve())})
    text_params = StdioServerParameters(command=sys.executable, args=['-m', 'toolatlas.text_tools_server'])
    async with Client(params) as memory, Client(text_params) as text:
        async def call(client, name, args):
            start = time.monotonic()
            result = await client.call_tool(name, args)
            assert not result.is_error, name
            records.append({'server': 'memory' if client is memory else 'text', 'tool': name,
                            'seconds': round(time.monotonic()-start, 6)})
            return _value(result)

        text_tools = (await text.list_tools()).tools
        memory_tools = (await memory.list_tools()).tools
        await call(memory, 'register_tools', {'tools': [dict(name=t.name, description=t.description,
            input_schema=t.input_schema, version='test', provider='local') for t in text_tools]})
        async def verify():
            normalized = await call(text, 'normalize_text', {'text': 'Hello, HELLO world!'})
            assert normalized == 'hello hello world'
            assert await call(text, 'word_count', {'text': normalized}) == 3
            assert await call(text, 'keyword_count', {'text': normalized, 'keyword': 'hello'}) == 2

        await verify()
        steps = [{'tool': 'normalize_text', 'rationale': 'normalize text'},
                 {'tool': 'keyword_count', 'rationale': 'count a keyword'}]
        await call(memory, 'remember_execution', dict(task_id='coverage', summary='normalize text count keyword',
            steps=steps, resolved=True, verifier_type='exact_match'))
        await verify()
        await call(memory, 'remember_rollouts', dict(task_id='coverage-batch', summary='normalize text count keyword',
            rollouts=[dict(steps=steps, resolved=True, verifier_type='exact_match')]))
        guide = await call(memory, 'get_guidance', {'task': 'normalize text count keyword'})
        assert guide['playbook']
        await call(memory, 'inspect_tool', {'tool_name': 'normalize_text'})
        await call(memory, 'suggest_probes', {'tool_name': 'normalize_text'})
        await call(memory, 'set_trace_status', {'task_id': 'coverage', 'status': 'quarantined', 'reason': 'governance smoke check'})
        await call(memory, 'refresh_status', {})
        await verify()  # external rerun must precede reactivation
        result = await call(memory, 'reverify_trace', {'task_id': 'coverage', 'resolved': True, 'verifier_type': 'exact_match'})
        assert result['status'] == 'active'
        await call(memory, 'memory_stats', {})
        for label, tools in [('memory', memory_tools), ('text', text_tools)]:
            called = {r['tool'] for r in records if r['server'] == label}
            assert called == {t.name for t in tools}, (label, called)
    sandbox = Path('mcp-sandbox/all-tools').resolve()
    sandbox.mkdir(parents=True)
    picture = sandbox / 'pixel.png'
    picture.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a5WQAAAAASUVORK5CYII='))
    params = StdioServerParameters(command='node', args=[str(filesystem_server_path(Path.cwd())), str(sandbox)])
    async with Client(params) as fs:
        tools = (await fs.list_tools()).tools
        file = str(sandbox / 'sample.txt')
        calls = [
            ('list_allowed_directories', {}),
            ('write_file', {'path': file, 'content': 'alpha beta'}),
            ('read_file', {'path': file}),
            ('read_text_file', {'path': file}),
            ('read_multiple_files', {'paths': [file]}),
            ('read_media_file', {'path': str(picture)}),
            ('edit_file', {'path': file, 'edits': [{'oldText': 'alpha', 'newText': 'gamma'}]}),
            ('create_directory', {'path': str(sandbox / 'moved')}),
            ('list_directory', {'path': str(sandbox)}),
            ('list_directory_with_sizes', {'path': str(sandbox)}),
            ('directory_tree', {'path': str(sandbox)}),
            ('search_files', {'path': str(sandbox), 'pattern': '*.txt'}),
            ('get_file_info', {'path': file}),
            ('move_file', {'source': file, 'destination': str(sandbox / 'moved/sample.txt')}),
        ]
        for name, args in calls:
            start = time.monotonic()
            result = await fs.call_tool(name, args)
            assert not result.is_error, name
            records.append({'server': 'filesystem', 'tool': name, 'seconds': round(time.monotonic()-start, 6)})
        assert (sandbox / 'moved/sample.txt').read_text() == 'gamma beta'
        assert not Path(file).exists()
        assert {name for name, _ in calls} == {t.name for t in tools}
        # Verify denial against a real file outside the provider allowlist.
        denied = False
        try:
            result = await fs.call_tool('read_text_file', {'path': str(Path('README.md').resolve())})
            denied = result.is_error
        except Exception as exc:
            denied = 'outside allowed' in str(exc).lower() or 'access denied' in str(exc).lower()
        assert denied, 'Filesystem boundary failed'
    print(json.dumps({'passed': True, 'filesystem_boundary_denied': denied, 'calls': records}, indent=2))


asyncio.run(main())
