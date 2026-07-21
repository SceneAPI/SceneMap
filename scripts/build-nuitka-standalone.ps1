param(
    [string] $PythonExecutable = "python",
    [string] $OutputDir = "dist/nuitka",
    [string] $OutputName = "sfmapi-colmap-api",
    [string] $ColmapBinDir = "../colmap-install-cuda-cudss/bin",
    [string[]] $RuntimeBinDirs = @("../vcpkg_installed_colmap_cuda/x64-windows/bin"),
    [switch] $NoColmapRuntime,
    [switch] $Clean
)

$ErrorActionPreference = "Stop"

function Invoke-NativeChecked {
    param(
        [string] $FilePath,
        [string[]] $ArgumentList
    )
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE"
    }
}

function Copy-RuntimeDirectory {
    param(
        [string] $Source,
        [string] $Destination
    )
    if (-not $Source -or -not (Test-Path $Source)) {
        return
    }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Copy-Item -Path (Join-Path $Source "*") -Destination $Destination -Recurse -Force
}

Invoke-NativeChecked $PythonExecutable @(
    "-c",
    "import sceneapi, scenemap.colmap, uvicorn, fastmcp; import nuitka"
)

if ($Clean -and (Test-Path $OutputDir)) {
    $resolvedOutput = Resolve-Path $OutputDir
    $repoRoot = Resolve-Path .
    if (-not $resolvedOutput.Path.StartsWith($repoRoot.Path)) {
        throw "Refusing to clean outside the repository: $($resolvedOutput.Path)"
    }
    Remove-Item -LiteralPath $resolvedOutput.Path -Recurse -Force
}

$nuitkaArgs = @(
    "-m", "nuitka",
    "--standalone",
    "--assume-yes-for-downloads",
    "--output-dir=$OutputDir",
    "--output-filename=$OutputName",
    "--include-package=sceneapi",
    "--include-package=scenemap.colmap",
    "--include-package=uvicorn",
    "--include-package=fastmcp",
    "--include-package=fastapi",
    "--include-package=starlette",
    "--include-package=pydantic",
    "--include-package=pydantic_settings",
    "--include-package=sqlalchemy",
    "--include-package=aiosqlite",
    "--include-package=alembic",
    "--include-package=anyio",
    "--include-package=sse_starlette",
    "--include-package=multipart",
    "--include-package=PIL",
    "--include-package=numpy",
    "--include-package=structlog",
    "--include-package=orjson",
    "--include-package=prometheus_client",
    "--include-package=httpx",
    "--include-module=uvicorn.loops.auto",
    "--include-module=uvicorn.protocols.http.auto",
    "--include-module=uvicorn.protocols.websockets.auto",
    "--include-module=uvicorn.lifespan.on",
    "--noinclude-pytest-mode=nofollow",
    "src/scenemap/colmap/api_launcher.py"
)

Invoke-NativeChecked $PythonExecutable $nuitkaArgs

$distDir = Join-Path $OutputDir "$OutputName.dist"
$actualDistDir = Join-Path $OutputDir "api_launcher.dist"
if ((Test-Path $actualDistDir) -and ($actualDistDir -ne $distDir)) {
    if (Test-Path $distDir) {
        Remove-Item -LiteralPath $distDir -Recurse -Force
    }
    Move-Item -LiteralPath $actualDistDir -Destination $distDir
}
if (-not (Test-Path $distDir)) {
    throw "Nuitka did not produce expected dist directory: $distDir"
}

if (-not $NoColmapRuntime) {
    $binDir = Join-Path $distDir "bin"
    Copy-RuntimeDirectory $ColmapBinDir $binDir
    foreach ($runtimeDir in $RuntimeBinDirs) {
        Copy-RuntimeDirectory $runtimeDir $binDir
    }

    $defaultCudssDirs = @(
        "C:/Program Files/NVIDIA cuDSS/v0.7/bin/13",
        "C:/Program Files/NVIDIA cuDSS/v0.7/bin/12"
    )
    foreach ($cudssDir in $defaultCudssDirs) {
        Copy-RuntimeDirectory $cudssDir $binDir
    }
}

$envExample = Join-Path $distDir "sfmapi-colmap-api.env.example"
@"
SFMAPI_BACKEND=colmap_cpp_native
SFMAPI_EPHEMERAL=true
SFMAPI_DB_URL=sqlite+aiosqlite:///file::memory:?cache=shared&uri=true
SFMAPI_BLOB_BACKEND=memory
SFMAPI_QUEUE_BACKEND=inline
SFMAPI_INLINE_TASKS=true
# Set to local to serve the MCP endpoint from this API process.
SFMAPI_MCP_MODE=off
SFMAPI_MCP_MOUNT_PATH=/mcp
"@ | Set-Content -Path $envExample -Encoding UTF8

Write-Host "Nuitka standalone bundle: $distDir"
Write-Host "Run: $distDir\$OutputName.exe --host 127.0.0.1 --port 8000"
Write-Host "MCP: $distDir\$OutputName.exe --mcp local --host 127.0.0.1 --port 8000"
