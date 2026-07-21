from __future__ import annotations

import os
from pathlib import Path

import pytest

from scenemap.realityscan.api_launcher import build_parser, configure_environment
from scenemap.realityscan.backend import (
    RealityScanCliBackend,
    configure_reality_cli_environment,
    configure_realityscan_environment,
    resolve_reality_cli_installation,
    resolve_realityscan_executable,
)


def _fake_realityscan(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def _exe_name() -> str:
    return "RealityScan.exe" if os.name == "nt" else "RealityScan"


def _clear_reality_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SFMAPI_RC_EXECUTABLE", raising=False)
    monkeypatch.delenv("SFMAPI_REALITYCAPTURE_EXECUTABLE", raising=False)
    monkeypatch.delenv("SFMAPI_REALITYSCAN_EXECUTABLE", raising=False)


def test_launcher_configures_realityscan_backend_from_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_exe = _fake_realityscan(tmp_path / "RealityScan" / _exe_name())
    monkeypatch.delenv("SFMAPI_BACKEND", raising=False)
    _clear_reality_env(monkeypatch)
    monkeypatch.setenv("PATH", "")

    args = build_parser().parse_args(
        [
            "--rc-executable",
            str(fake_exe),
            "--mcp",
            "local",
            "--mcp-mount-path",
            "/agent",
        ]
    )
    selected = configure_environment(args)

    assert selected == fake_exe.resolve()
    assert os.environ["SFMAPI_BACKEND"] == "realityscan_cli"
    assert os.environ["SFMAPI_RC_EXECUTABLE"] == str(fake_exe.resolve())
    assert os.environ["SFMAPI_REALITYSCAN_EXECUTABLE"] == str(fake_exe.resolve())
    assert os.environ["SFMAPI_EPHEMERAL"] == "true"
    assert os.environ["SFMAPI_BLOB_BACKEND"] == "memory"
    assert os.environ["SFMAPI_MCP_MODE"] == "local"
    assert os.environ["SFMAPI_MCP_MOUNT_PATH"] == "/agent"
    assert str(fake_exe.parent.resolve()) in os.environ["PATH"].split(os.pathsep)


def test_realityscan_environment_accepts_install_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_dir = tmp_path / "RealityScan"
    fake_exe = _fake_realityscan(install_dir / _exe_name())
    _clear_reality_env(monkeypatch)
    monkeypatch.setenv("PATH", "")

    selected = configure_realityscan_environment(install_dir, validate=True)

    assert selected == fake_exe.resolve()
    assert RealityScanCliBackend().runtime_versions()["realityscan_executable"] == str(
        fake_exe.resolve()
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows install layout")
def test_realityscan_environment_detects_versioned_epic_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    older_exe = _fake_realityscan(
        tmp_path / "Program Files" / "Epic Games" / "RealityScan_1.9" / "RealityScan.exe"
    )
    assert older_exe.exists()
    fake_exe = _fake_realityscan(
        tmp_path / "Program Files" / "Epic Games" / "RealityScan_2.1" / "RealityScan.exe"
    )
    _clear_reality_env(monkeypatch)
    monkeypatch.setenv("PROGRAMFILES", str(tmp_path / "Program Files"))
    monkeypatch.delenv("PROGRAMFILES(X86)", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setenv("PATH", "")

    selected = configure_realityscan_environment()

    assert selected == fake_exe.resolve()
    assert os.environ["SFMAPI_REALITYSCAN_EXECUTABLE"] == str(fake_exe.resolve())


@pytest.mark.skipif(os.name != "nt", reason="Windows install layout")
def test_realityscan_environment_prefers_system_realitycapture_over_realityscan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    program_files = tmp_path / "Program Files"
    rc_exe = _fake_realityscan(
        program_files / "Epic Games" / "RealityCapture_3.0" / "RealityCapture.exe"
    )
    scan_exe = _fake_realityscan(
        program_files / "Epic Games" / "RealityScan_9.9" / "RealityScan.exe"
    )
    assert scan_exe.exists()
    _clear_reality_env(monkeypatch)
    monkeypatch.setenv("PROGRAMFILES", str(program_files))
    monkeypatch.delenv("PROGRAMFILES(X86)", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setenv("PATH", "")

    installation = configure_reality_cli_environment()

    assert installation is not None
    assert installation.executable == rc_exe.resolve()
    assert installation.interface.interface_id == "realitycapture.current"
    assert os.environ["SFMAPI_REALITYCAPTURE_EXECUTABLE"] == str(rc_exe.resolve())


def test_explicit_rc_env_wins_over_default_detection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    explicit = _fake_realityscan(tmp_path / "Explicit" / _exe_name())
    _clear_reality_env(monkeypatch)
    monkeypatch.setenv("SFMAPI_RC_EXECUTABLE", str(explicit))
    monkeypatch.setenv("PATH", "")

    selected = configure_realityscan_environment()

    assert selected == explicit.resolve()


def test_invalid_configured_realityscan_path_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="RealityCapture/RealityScan executable not found"):
        configure_realityscan_environment(tmp_path / "missing" / _exe_name(), validate=True)


def test_resolve_realityscan_executable_accepts_direct_file(tmp_path: Path) -> None:
    fake_exe = _fake_realityscan(tmp_path / _exe_name())

    assert resolve_realityscan_executable(fake_exe) == fake_exe.resolve()
    installation = resolve_reality_cli_installation(fake_exe)
    assert installation is not None
    assert installation.interface.product == "realityscan"
