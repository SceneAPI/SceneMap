param(
    [string]$SphereSfMExecutable = $env:SFMAPI_SPHERESFM_EXECUTABLE,
    [int]$Port = 8000
)

uv run sfmapi-spheresfm-api --spheresfm-executable $SphereSfMExecutable --port $Port --mcp local

