from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from scenemap.colmap.native.backend import COLMAP_COMMANDS

# From sfmapi_colmap's test_colmap_native_contract.py. The shared
# submodule test moved to test_submodule_contract.py; the build scripts
# keep their names (scripts/build-colmap.*).

REPO_ROOT = Path(__file__).resolve().parents[2]
OPTIONAL_DISABLED_COMMANDS = {"delaunay_mesher": "requires CGAL"}


def test_build_scripts_target_the_upstream_submodule():
    ps1 = (REPO_ROOT / "scripts" / "build-colmap.ps1").read_text(encoding="utf-8")
    sh = (REPO_ROOT / "scripts" / "build-colmap.sh").read_text(encoding="utf-8")

    for text in (ps1, sh):
        assert "git submodule update --init --recursive" in text
        assert "third_party/colmap" in text
        assert "CUDA_ENABLED" in text
        assert "GUI_ENABLED" in text
        assert "cudss_DIR" in text
        assert "CGAL_ENABLED" in text
        assert "GFLAGS_USE_TARGET_NAMESPACE" in text
        assert "OpenMP_CUDA_FLAGS" in text
        assert "CERES_STATIC_DEFINE" in text
        assert "cgal" in text.lower()
        assert "vcpkg" in text.lower()
        assert "CMAKE_PREFIX_PATH" in text
        assert "--build" in text
        assert "--install" in text
        assert "pip" in text
        assert "install" in text

    assert "$CmakePrefixPath;$vcpkgPrefix" in ps1
    assert "[switch] $Gui" in ps1
    assert '"OFF" }' in ps1
    assert "${cmake_prefix_path};${vcpkg_prefix}" in sh
    assert "SFMAPI_COLMAP_GUI:-OFF" in sh


@pytest.mark.integration
@pytest.mark.needs_colmap
def test_colmap_executable_help_lists_expected_cli_surface(colmap_executable: Path):
    result = subprocess.run(
        [str(colmap_executable), "-h"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    text = result.stdout + result.stderr

    assert "COLMAP" in text
    assert "Available commands" in text
    for command in COLMAP_COMMANDS:
        assert command in text
    assert "gui" not in COLMAP_COMMANDS


@pytest.mark.integration
@pytest.mark.needs_colmap
@pytest.mark.parametrize("command", COLMAP_COMMANDS)
def test_every_generic_bridge_command_has_native_help(colmap_executable: Path, command: str):
    result = subprocess.run(
        [str(colmap_executable), command, "-h"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    text = result.stdout + result.stderr

    optional_reason = OPTIONAL_DISABLED_COMMANDS.get(command)
    if result.returncode != 0 and optional_reason and optional_reason in text:
        return

    assert result.returncode == 0
    assert "COLMAP" in text
    if command == "version":
        assert "Commit" in text
        return
    assert "-h [ --help ]" in text
    assert any(part.startswith("--") for part in text.replace(",", " ").split())


@pytest.mark.integration
@pytest.mark.needs_colmap
@pytest.mark.skipif(
    os.environ.get("SFMAPI_COLMAP_RUN_BUILD_TEST") != "1",
    reason="native COLMAP build validation is expensive; set SFMAPI_COLMAP_RUN_BUILD_TEST=1",
)
def test_native_build_script_can_configure_and_build_colmap(tmp_path: Path):
    build_dir = tmp_path / "colmap-build"
    install_prefix = tmp_path / "colmap-install"
    extra_args: list[str] = ["-InstallPrefix", str(install_prefix), "-NoOnnx"]
    project_root = REPO_ROOT.parent
    vcpkg_prefix = project_root / "vcpkg_installed_colmap_cuda" / "x64-windows"
    ceres_prefix = project_root / "ceres-install-cuda-cudss"
    ceres_dir = ceres_prefix / "lib" / "cmake" / "Ceres"
    cmake_prefix_parts = []
    if ceres_prefix.exists():
        cmake_prefix_parts.append(str(ceres_prefix))
    if vcpkg_prefix.exists():
        cmake_prefix_parts.append(str(vcpkg_prefix))
    if cmake_prefix_parts:
        extra_args.extend(["-CmakePrefixPath", ";".join(cmake_prefix_parts)])
    if ceres_dir.exists():
        extra_args.extend(["-CeresDir", str(ceres_dir), "-StaticCeres"])
    cudss_env = os.environ.get("SFMAPI_COLMAP_CUDSS_DIR")
    cudss_dir = (
        Path(cudss_env)
        if cudss_env
        else Path("C:/Program Files/NVIDIA cuDSS/v0.7/lib/13/cmake/cudss")
    )
    if cudss_dir.exists():
        extra_args.extend(["-CudssDir", str(cudss_dir)])
    cuda_root = os.environ.get("CUDA_PATH")
    if cuda_root and Path(cuda_root).exists():
        extra_args.extend(["-Cuda", "-CudaToolkitRoot", cuda_root, "-CudaArchitectures", "native"])

    if os.name == "nt":
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell is None:
            pytest.skip("PowerShell executable not found")
        subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(REPO_ROOT / "scripts" / "build-colmap.ps1"),
                "-Config",
                "Release",
                "-BuildDir",
                str(build_dir),
                *extra_args,
            ],
            cwd=REPO_ROOT,
            check=True,
            timeout=3600,
        )
    else:
        subprocess.run(
            [
                "bash",
                str(REPO_ROOT / "scripts" / "build-colmap.sh"),
                "Release",
                str(build_dir),
            ],
            cwd=REPO_ROOT,
            env={
                **os.environ,
                "SFMAPI_COLMAP_INSTALL_PREFIX": str(install_prefix),
                "SFMAPI_COLMAP_GUI": "OFF",
                "SFMAPI_COLMAP_ONNX": "OFF",
                "SFMAPI_COLMAP_CMAKE_PREFIX_PATH": ";".join(cmake_prefix_parts),
                "SFMAPI_COLMAP_CERES_DIR": str(ceres_dir) if ceres_dir.exists() else "",
                "SFMAPI_COLMAP_STATIC_CERES": "1" if ceres_dir.exists() else "0",
                "SFMAPI_COLMAP_CUDSS_DIR": str(cudss_dir) if cudss_dir.exists() else "",
                "SFMAPI_COLMAP_CUDA": "ON" if cuda_root and Path(cuda_root).exists() else "OFF",
                "SFMAPI_COLMAP_CUDATOOLKIT_ROOT": cuda_root or "",
            },
            check=True,
            timeout=3600,
        )
