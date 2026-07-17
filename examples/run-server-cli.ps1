$ErrorActionPreference = "Stop"

param(
    [string] $ColmapExecutable = $env:SFMAPI_COLMAP_EXECUTABLE,
    [int] $Port = 8000,
    [switch] $Reload,
    [switch] $Mcp
)

if (-not $ColmapExecutable) {
    throw "Set -ColmapExecutable or SFMAPI_COLMAP_EXECUTABLE to the path of colmap.exe."
}

$args = @(
    "run",
    "sfmapi-colmap-cli-api",
    "--colmap-executable",
    $ColmapExecutable,
    "--port",
    "$Port"
)
if ($Reload) {
    $args += "--reload"
}
if ($Mcp) {
    $args += @("--mcp", "local")
}

uv @args
