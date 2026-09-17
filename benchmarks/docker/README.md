# Isolated ToolAtlas validation

Run from PowerShell:

```powershell
powershell -NoProfile -File benchmarks/docker/run.ps1
```

The image uses the repository's pinned official Node MCP packages and installs
Python dependencies in `.venv`. The build context is allowlisted: `.env`, local
databases, Git history, and existing results never enter the image. Build-time
network access installs dependencies. Runtime uses UID 10001, no network, a
read-only root filesystem, dropped capabilities, no-new-privileges, a PID limit,
2 CPUs and 2 GiB RAM. No host bind mount or Docker socket is exposed. An isolated
Docker volume holds disposable fixtures and output; the script copies results
out and removes its container and volume even when checks fail.

Checks exercise actual stdio servers, all discovered local memory/text tools,
all 14 Filesystem tools (including writes and moves within disposable data),
all discovered Everything tools, the full pytest suite, demos, and both
four-attempt deterministic benchmark splits. Tool coverage records per-call
elapsed time; those timings are smoke measurements, not a latency benchmark.
GitHub tests explicitly skip without credentials. The runner never loads `.env`
or silently starts paid LLM requests.

The image also downloads MCPMark's official `file_property` snapshot at build
time with a pinned SHA-256. A custom deterministic policy performs the
`size_classification` task through Filesystem MCP on four fresh copies, then
runs the unmodified verifier from the digest-pinned MCPMark image. This tests
snapshot, write-tool, and verifier compatibility. It has no ToolAtlas-assisted
arm and no independent LLM samples; do not publish it as an MCPMark model score.

The deterministic control is not a reproduction of the paper. In particular,
the paper's cross-environment protocol partitions environment instances about
2:3; this repository uses two small fixture environments with three training
and six test tasks. Its fixed baseline includes discovery calls, so savings
measure structural call elimination. Report provider and total MCP calls
together. Repeated deterministic attempts are not independent LLM samples.

The full paper additionally needs official task snapshots and verifiers,
LLM-backed baseline and assisted arms, frozen trained memory, inference-token
accounting, cross-agent transfer, and ablations. Docker isolation does not
isolate the remote account behind an API credential: GitHub and Notion writes
need dedicated disposable benchmark resources.
