$ErrorActionPreference = "Stop"

$env:SFMAPI_EPHEMERAL = "true"
$env:SFMAPI_BACKEND = "colmap_cli"
$env:SFMAPI_MCP_MODE = "off"

if (-not $env:SFMAPI_COLMAP_EXECUTABLE) {
    $candidates = @(
        ".\third_party\colmap\build\src\colmap\exe\Release\colmap.exe",
        ".\third_party\colmap\build\src\colmap\exe\colmap.exe",
        ".\third_party\colmap\build\src\exe\colmap.exe",
        ".\third_party\colmap\build\bin\colmap.exe",
        ".\third_party\colmap\build\colmap.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            $env:SFMAPI_COLMAP_EXECUTABLE = (Resolve-Path $candidate)
            break
        }
    }
}

uv run sfmapi-colmap-api --backend colmap_cli --reload
