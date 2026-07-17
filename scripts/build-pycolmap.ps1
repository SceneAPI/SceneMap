param(
    [string] $Config = "Release",
    [string] $BuildDir = "third_party/colmap/build",
    [string] $InstallPrefix = "third_party/colmap/install",
    [switch] $Cuda,
    [string] $CudaToolkitRoot = "",
    [string] $CudaArchitectures = "native",
    [string] $CmakePrefixPath = "",
    [string] $CeresDir = "",
    [switch] $BuildPycolmap,
    [string] $PythonExecutable = "python",
    [switch] $NoGui,
    [switch] $NoOnnx,
    [switch] $Caspar
)

$ErrorActionPreference = "Stop"

git submodule update --init --recursive

$cudaEnabled = if ($Cuda) { "ON" } else { "OFF" }
$guiEnabled = if ($NoGui) { "OFF" } else { "ON" }
$onnxEnabled = if ($NoOnnx) { "OFF" } else { "ON" }
$casparEnabled = if ($Caspar) { "ON" } else { "OFF" }

$cmakeArgs = @(
    "-S", "third_party/colmap",
    "-B", $BuildDir,
    "-DCMAKE_BUILD_TYPE=$Config",
    "-DCMAKE_INSTALL_PREFIX=$InstallPrefix",
    "-DCUDA_ENABLED=$cudaEnabled",
    "-DGUI_ENABLED=$guiEnabled",
    "-DONNX_ENABLED=$onnxEnabled",
    "-DCASPAR_ENABLED=$casparEnabled"
)

if ($Cuda) {
    $cmakeArgs += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
    if ($CudaToolkitRoot) {
        $cmakeArgs += "-DCUDAToolkit_ROOT=$CudaToolkitRoot"
    }
}
if ($CmakePrefixPath) {
    $cmakeArgs += "-DCMAKE_PREFIX_PATH=$CmakePrefixPath"
}
if ($CeresDir) {
    $cmakeArgs += "-DCeres_DIR=$CeresDir"
}

cmake @cmakeArgs
cmake --build $BuildDir --config $Config --parallel
cmake --install $BuildDir --config $Config

if ($BuildPycolmap) {
    $oldPrefixPath = $env:CMAKE_PREFIX_PATH
    $env:CMAKE_PREFIX_PATH = "$((Resolve-Path $InstallPrefix).Path);$oldPrefixPath"
    try {
        & $PythonExecutable -m pip install third_party/colmap --force-reinstall
    }
    finally {
        $env:CMAKE_PREFIX_PATH = $oldPrefixPath
    }
}

Write-Host "COLMAP build finished under $BuildDir"
Write-Host "COLMAP install prefix: $InstallPrefix"
