from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Deduplicated from sfmapi_colmap / sfmapi_pycolmap (identical copies).
# Merge adaptation: this repo registers the third_party/colmap gitlink
# without checking its contents out (see README), so the materialized
# checks skip until `git submodule update --init third_party/colmap`
# has been run — the sources asserted an always-initialized submodule.


def test_colmap_submodule_is_configured() -> None:
    gitmodules = (REPO_ROOT / ".gitmodules").read_text(encoding="utf-8")

    assert "path = third_party/colmap" in gitmodules
    assert "url = https://github.com/colmap/colmap.git" in gitmodules
    assert "branch = main" in gitmodules


def test_colmap_submodule_gitlink_pins_expected_commit() -> None:
    result = subprocess.run(
        ["git", "ls-files", "--stage", "third_party/colmap"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        pytest.skip(f"git metadata unavailable: {result.stderr.strip()}")
    mode, sha = result.stdout.split()[:2]
    assert mode == "160000"
    # Same upstream commit the three superseded repos pinned.
    assert sha == "6cfbc0404cb973ca79a6835bb88bb020ed96bc8f"


def test_colmap_submodule_contents_when_initialized() -> None:
    if not (REPO_ROOT / "third_party" / "colmap" / "CMakeLists.txt").is_file():
        pytest.skip(
            "third_party/colmap not initialized; run "
            "`git submodule update --init third_party/colmap`"
        )

    result = subprocess.run(
        ["git", "submodule", "status", "third_party/colmap"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"git worktree metadata unavailable: {result.stderr.strip()}")
    assert result.stdout.strip()
    assert not result.stdout.lstrip().startswith("-")
