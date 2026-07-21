from __future__ import annotations

import os
from pathlib import Path

from scenemap.instantsfm.api_launcher import build_parser, configure_environment


def _fake_instantsfm(root: Path) -> Path:
    (root / "instantsfm").mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname='instantsfm'\n", encoding="utf-8")
    return root


def test_launcher_configures_in_memory_sfmapi(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = _fake_instantsfm(tmp_path / "InstantSfM")
    parser = build_parser()
    args = parser.parse_args(["--instantsfm-root", str(root), "--mcp", "local"])

    selected = configure_environment(args)

    assert selected == root.resolve()
    assert os.environ["SFMAPI_BACKEND"] == "instantsfm"
    assert os.environ["SFMAPI_BLOB_BACKEND"] == "memory"
    assert os.environ["SFMAPI_QUEUE_BACKEND"] == "inline"
    assert os.environ["SFMAPI_INLINE_TASKS"] == "true"
    assert os.environ["SFMAPI_MCP_MODE"] == "local"


def test_launcher_dry_run_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["--dry-run"])

    assert args.dry_run is True
