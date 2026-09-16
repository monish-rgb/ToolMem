# MCPMark Filesystem Smoke Result

This is an official MCPMark Docker smoke run for the Filesystem task
`file_property/size_classification`.

- Date: 2026-09-16
- MCPMark commit: `cd45b7f57923b9b3985467f5139927575f83141c`
- Docker image: `evalsysorg/mcpmark@sha256:05379ddb62790d9037a5521c46155b92304d95ae192caa76edfc5b8d2ac5474f`
- Model: `nvidia-kimi-k3` through an NVIDIA OpenAI-compatible endpoint
- Command shape: `python3 -m pipeline --mcp filesystem --models nvidia-kimi-k3 --exp-name toolatlas-mcpmark-smoke --tasks file_property/size_classification --k 1 --timeout 900`

## Result

- Passed: 1/1 tasks, 100.0%
- Verifier: passed all file-classification checks
- Agent execution time: 667.41 seconds
- Total task time: 673.50 seconds
- Turns: 6
- Tokens: 18,855 input, 1,326 output, 20,181 total
- MCP tool calls: 15 total

Tool-call breakdown:

| Tool | Calls |
| --- | ---: |
| `list_allowed_directories` | 1 |
| `list_directory` | 1 |
| `list_directory_with_sizes` | 1 |
| `create_directory` | 3 |
| `move_file` | 9 |

## Interpretation

This run proves that the official MCPMark Docker benchmark path works locally
with the Filesystem MCP server, the NVIDIA-backed model route, and MCPMark's
programmatic verifier.

It is intentionally labeled as a baseline smoke result. It does not yet prove
ToolAtlas reduces calls on MCPMark. To make that claim, run a matched
ToolAtlas-assisted arm on the same MCPMark task set, keep memory frozen after
training, use the paper-style `k=4` independent rollouts, and compare verifier
pass rate, turns, tool calls, and token usage.
