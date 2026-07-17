$ErrorActionPreference = "Stop"

param(
    [string] $RcExecutable = $env:SFMAPI_RC_EXECUTABLE,
    [int] $Port = 8000,
    [switch] $Reload,
    [switch] $Mcp
)

if (-not $RcExecutable) {
    $RcExecutable = $env:SFMAPI_REALITYCAPTURE_EXECUTABLE
}
if (-not $RcExecutable) {
    $RcExecutable = $env:SFMAPI_REALITYSCAN_EXECUTABLE
}

$args = @(
    "run",
    "sfmapi-realityscan-api",
    "--port",
    "$Port"
)
if ($RcExecutable) {
    $args += @("--rc-executable", $RcExecutable)
}
if ($Reload) {
    $args += "--reload"
}
if ($Mcp) {
    $args += @("--mcp", "local")
}

uv @args
