from __future__ import annotations

import os
from pathlib import Path

from sceneapi_map.spheresfm.api_launcher import build_parser, configure_environment


def _fake_colmap(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def test_launcher_configures_in_memory_sfmapi(tmp_path: Path) -> None:
    exe = _fake_colmap(tmp_path / "build" / "src" / "exe" / "colmap.exe")
    parser = build_parser()
    args = parser.parse_args(["--spheresfm-executable", str(exe), "--mcp", "local"])

    selected = configure_environment(args)

    assert selected == exe.resolve()
    assert os.environ["SFMAPI_BACKEND"] == "spheresfm"
    assert os.environ["SFMAPI_BLOB_BACKEND"] == "memory"
    assert os.environ["SFMAPI_QUEUE_BACKEND"] == "inline"
    assert os.environ["SFMAPI_INLINE_TASKS"] == "true"
    assert os.environ["SFMAPI_MCP_MODE"] == "local"


def test_launcher_dry_run_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["--dry-run"])

    assert args.dry_run is True
