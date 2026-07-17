$ErrorActionPreference = "Stop"

param(
    [string] $ColmapExecutable = $env:SFMAPI_COLMAP_EXECUTABLE,
    [int] $Port = 8000,
    [switch] $Reload,
    [switch] $Mcp
)

$args = @(
    "run",
    "sfmapi-pycolmap-api",
    "--backend",
    "colmap_pycolmap",
    "--port",
    "$Port"
)
if ($ColmapExecutable) {
    $args += @("--colmap-executable", $ColmapExecutable)
}
if ($Reload) {
    $args += "--reload"
}
if ($Mcp) {
    $args += @("--mcp", "local")
}

uv @args
