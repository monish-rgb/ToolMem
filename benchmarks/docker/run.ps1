$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$name = 'toolatlas-benchmark-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
$output = Join-Path $root "benchmarks/results/$name"
docker build -f (Join-Path $PSScriptRoot 'Dockerfile') -t toolatlas-secure-benchmark $root
if ($LASTEXITCODE -ne 0) { throw 'Docker build failed' }
# A named volume preserves output for docker cp after container exit; no host bind mount.
$volume = "$name-work"
docker volume create $volume | Out-Null
try {
    docker create --name $name --network none --read-only --cap-drop ALL --security-opt no-new-privileges --pids-limit 256 --memory 2g --cpus 2 --tmpfs /tmp:rw,noexec,nosuid,size=256m --mount "type=volume,src=$volume,dst=/work" toolatlas-secure-benchmark | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Docker create failed' }
    # Docker initializes the volume using the image directory ownership.
    docker start -a $name
    $runExit = $LASTEXITCODE
    New-Item -ItemType Directory -Path $output -Force | Out-Null
    docker cp "${name}:/work/results/." $output
    if ($LASTEXITCODE -ne 0) { throw 'Failed to copy benchmark results' }
    docker inspect --format '{{json .HostConfig}}' $name | Set-Content (Join-Path $output 'container-security.json')
    Write-Output "Results: $output"
    if ($runExit -ne 0) { throw "Benchmark runner failed: $runExit; inspect saved logs" }
} finally {
    docker rm -f $name | Out-Null
    docker volume rm $volume | Out-Null
}
