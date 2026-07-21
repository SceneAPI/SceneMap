from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from scenemap.colmap.pycolmap.backend import COLMAP_COMMANDS

# From sfmapi_pycolmap's test_colmap_native_contract.py. The shared
# submodule test moved to test_submodule_contract.py; that repo's
# build-colmap.* scripts (which build pycolmap from the submodule) are
# carried as scripts/build-pycolmap.* to coexist with the native
# COLMAP-executable build scripts.

REPO_ROOT = Path(__file__).resolve().parents[2]
OPTIONAL_DISABLED_COMMANDS = {"delaunay_mesher": "requires CGAL"}


def test_build_scripts_target_the_upstream_submodule():
    ps1 = (REPO_ROOT / "scripts" / "build-pycolmap.ps1").read_text(encoding="utf-8")
    sh = (REPO_ROOT / "scripts" / "build-pycolmap.sh").read_text(encoding="utf-8")

    for text in (ps1, sh):
        assert "git submodule update --init --recursive" in text
        assert "third_party/colmap" in text
        assert "CUDA_ENABLED" in text
        assert "CMAKE_PREFIX_PATH" in text
        assert "cmake --build" in text
        assert "pip install third_party/colmap" in text


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
    assert "gui" in text


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
    if os.name == "nt":
        subprocess.run(
            [
                "pwsh",
                str(REPO_ROOT / "scripts" / "build-pycolmap.ps1"),
                "-Config",
                "Release",
                "-BuildDir",
                str(build_dir),
            ],
            cwd=REPO_ROOT,
            check=True,
            timeout=3600,
        )
    else:
        subprocess.run(
            [
                "bash",
                str(REPO_ROOT / "scripts" / "build-pycolmap.sh"),
                "Release",
                str(build_dir),
            ],
            cwd=REPO_ROOT,
            check=True,
            timeout=3600,
        )
