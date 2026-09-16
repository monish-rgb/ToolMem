# ToolAtlas paper-protocol Filesystem benchmark

Generated: 2026-09-16T15:37:28.371133+00:00

This is a deterministic, read-only Filesystem control using the evaluation shape from the ToolAtlas paper. It is not the full MCPMark/MCP-Universe reproduction and does not establish production readiness.

| Split | Arm | pass@1 | pass@4 | Provider calls | Total MCP calls |
|---|---|---:|---:|---:|---:|
| same_environment | baseline | 1.0000 | 1.0000 | 96 | 96 |
| same_environment | toolatlas | 1.0000 | 1.0000 | 48 | 72 |

- same_environment provider-call reduction: 50.00% (48 calls).
- same_environment total-MCP-call reduction: 25.00% (24 calls, including one guidance call per assisted run).
- Training-cost break-even: 10 evaluation runs.
| cross_environment | baseline | 1.0000 | 1.0000 | 96 | 96 |
| cross_environment | toolatlas | 1.0000 | 1.0000 | 48 | 72 |

- cross_environment provider-call reduction: 50.00% (48 calls).
- cross_environment total-MCP-call reduction: 25.00% (24 calls, including one guidance call per assisted run).
- Training-cost break-even: 10 evaluation runs.

## Interpretation

The control demonstrates that retrieved provider memory removes redundant Filesystem discovery calls while preserving exact verifier success. Because the runner is deterministic and the task family is small, use the optional LLM harness and official benchmark adapters before making production claims.
