param(
    [string] $Config = "Release",
    [string] $BuildDir = "third_party/colmap/build",
    [string] $InstallPrefix = "third_party/colmap/install",
    [switch] $Cuda,
    [string] $CudaToolkitRoot = "",
    [string] $CudaArchitectures = "native",
    [string] $OpenMPCudaFlags = "",
    [string] $CmakePrefixPath = "",
    [string] $CeresDir = "",
    [switch] $StaticCeres,
    [string] $CudssDir = "",
    [switch] $NoCgal,
    [string] $CgalDir = "",
    [switch] $InstallCgal,
    [string] $VcpkgRoot = "../vcpkg",
    [string] $VcpkgTriplet = "x64-windows",
    [string] $VcpkgInstallRoot = "../vcpkg_installed_colmap_cuda",
    [switch] $BuildPycolmap,
    [string] $PythonExecutable = "python",
    [switch] $Gui,
    [switch] $NoGui,
    [switch] $NoOnnx,
    [switch] $Caspar
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

function Resolve-CmakePrefixPath {
    param([string] $Value)
    if (-not $Value) {
        return ""
    }
    $resolvedParts = @()
    foreach ($part in ($Value -split ";")) {
        if (-not $part) {
            continue
        }
        if (Test-Path $part) {
            $resolvedParts += (Resolve-Path $part).Path
        }
        else {
            $resolvedParts += $part
        }
    }
    return ($resolvedParts -join ";")
}

git submodule update --init --recursive

$cudaEnabled = if ($Cuda) { "ON" } else { "OFF" }
$cgalEnabled = if ($NoCgal) { "OFF" } else { "ON" }
$guiEnabled = if ($Gui -and -not $NoGui) { "ON" } else { "OFF" }
$onnxEnabled = if ($NoOnnx) { "OFF" } else { "ON" }
$casparEnabled = if ($Caspar) { "ON" } else { "OFF" }

if ($Cuda -and -not $OpenMPCudaFlags) {
    $OpenMPCudaFlags = "-Xcompiler=/openmp"
}

if ($InstallCgal) {
    $vcpkgExe = Join-Path $VcpkgRoot "vcpkg.exe"
    if (-not (Test-Path $vcpkgExe)) {
        $vcpkgExe = Join-Path $VcpkgRoot "vcpkg"
    }
    if (-not (Test-Path $vcpkgExe)) {
        throw "vcpkg executable not found under $VcpkgRoot"
    }
    Invoke-NativeChecked $vcpkgExe @("install", "cgal:$VcpkgTriplet", "--x-install-root=$VcpkgInstallRoot", "--recurse")
    $vcpkgPrefix = (Resolve-Path (Join-Path $VcpkgInstallRoot $VcpkgTriplet)).Path
    $CmakePrefixPath = if ($CmakePrefixPath) { "$CmakePrefixPath;$vcpkgPrefix" } else { $vcpkgPrefix }
}

$CmakePrefixPath = Resolve-CmakePrefixPath $CmakePrefixPath
if ($CeresDir -and (Test-Path $CeresDir)) {
    $CeresDir = (Resolve-Path $CeresDir).Path
}
if ($CgalDir -and (Test-Path $CgalDir)) {
    $CgalDir = (Resolve-Path $CgalDir).Path
}
if ($CudssDir -and (Test-Path $CudssDir)) {
    $CudssDir = (Resolve-Path $CudssDir).Path
}

$cmakeArgs = @(
    "-S", "third_party/colmap",
    "-B", $BuildDir,
    "-DCMAKE_BUILD_TYPE=$Config",
    "-DCMAKE_INSTALL_PREFIX=$InstallPrefix",
    "-DCUDA_ENABLED=$cudaEnabled",
    "-DCGAL_ENABLED=$cgalEnabled",
    "-DGUI_ENABLED=$guiEnabled",
    "-DONNX_ENABLED=$onnxEnabled",
    "-DCASPAR_ENABLED=$casparEnabled",
    "-DGFLAGS_USE_TARGET_NAMESPACE=ON"
)

if ($Cuda) {
    $cmakeArgs += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
    if ($CudaToolkitRoot) {
        $cmakeArgs += "-DCUDAToolkit_ROOT=$CudaToolkitRoot"
    }
    if ($OpenMPCudaFlags) {
        $cmakeArgs += "-DOpenMP_CUDA_FLAGS=$OpenMPCudaFlags"
        $cmakeArgs += "-DOpenMP_CUDA_LIB_NAMES="
    }
}
if ($CmakePrefixPath) {
    $cmakeArgs += "-DCMAKE_PREFIX_PATH=$CmakePrefixPath"
}
if ($CeresDir) {
    $cmakeArgs += "-DCeres_DIR=$CeresDir"
}
if ($CudssDir) {
    $cmakeArgs += "-Dcudss_DIR=$CudssDir"
}
if ($CgalDir) {
    $cmakeArgs += "-DCGAL_DIR=$CgalDir"
}
if ($StaticCeres) {
    $cmakeArgs += "-DCMAKE_CXX_FLAGS=/DCERES_STATIC_DEFINE"
    $cmakeArgs += "-DCMAKE_CUDA_FLAGS=-DCERES_STATIC_DEFINE"
}

Invoke-NativeChecked "cmake" $cmakeArgs
Invoke-NativeChecked "cmake" @("--build", $BuildDir, "--config", $Config, "--parallel")
Invoke-NativeChecked "cmake" @("--install", $BuildDir, "--config", $Config)

if ($BuildPycolmap) {
    $oldPrefixPath = $env:CMAKE_PREFIX_PATH
    $env:CMAKE_PREFIX_PATH = "$((Resolve-Path $InstallPrefix).Path);$oldPrefixPath"
    try {
        Invoke-NativeChecked $PythonExecutable @("-m", "pip", "install", "third_party/colmap", "--force-reinstall")
    }
    finally {
        $env:CMAKE_PREFIX_PATH = $oldPrefixPath
    }
}

Write-Host "COLMAP build finished under $BuildDir"
Write-Host "COLMAP install prefix: $InstallPrefix"
