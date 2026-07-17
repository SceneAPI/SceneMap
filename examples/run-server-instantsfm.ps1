param(
    [string]$InstantSfMRoot = ".\third_party\instantsfm",
    [int]$Port = 8000
)

uv run sfmapi-instantsfm-api --instantsfm-root $InstantSfMRoot --port $Port --mcp local

