from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from scenemap.colmap import api_launcher
from scenemap.colmap.cli.backend import (
    ColmapCliBackend,
    configure_colmap_environment,
    resolve_colmap_executable,
)

# Union of the three superseded repos' launcher suites, adapted to the
# provider-parameterized api_launcher (parse_args/build_parser take the
# provider; configure_environment reads it from args).


def _fake_colmap(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def _exe_name() -> str:
    return "colmap.exe" if os.name == "nt" else "colmap"


# ---- native launcher (ex sfmapi-colmap) ---------------------------------


def test_native_launcher_configures_api_environment(monkeypatch):
    for key in (
        "SFMAPI_BACKEND",
        "SFMAPI_COLMAP_EXECUTABLE",
        "SFMAPI_DB_URL",
        "SFMAPI_EPHEMERAL",
        "SFMAPI_MCP_MODE",
        "SFMAPI_MCP_MOUNT_PATH",
    ):
        monkeypatch.delenv(key, raising=False)

    args = api_launcher.parse_args(
        [
            "--backend",
            "colmap_cpp_inmemory",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--colmap-executable",
            "colmap.exe",
            "--mcp",
            "local",
            "--mcp-mount-path",
            "/agent",
        ],
        provider="native",
    )
    api_launcher.configure_environment(args)

    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert os.environ["SFMAPI_BACKEND"] == "colmap_cpp_inmemory"
    assert os.environ["SFMAPI_EPHEMERAL"] == "true"
    assert os.environ["SFMAPI_DB_URL"].startswith("sqlite+aiosqlite:///")
    assert os.environ["SFMAPI_COLMAP_EXECUTABLE"].endswith("colmap.exe")
    assert os.environ["SFMAPI_MCP_MODE"] == "local"
    assert os.environ["SFMAPI_MCP_MOUNT_PATH"] == "/agent"


def test_native_launcher_detects_bundled_colmap_executable(tmp_path: Path, monkeypatch):
    dist_dir = tmp_path / "sfmapi-colmap-api.dist"
    bundled = dist_dir / "bin" / _exe_name()
    bundled.parent.mkdir(parents=True)
    bundled.write_text("", encoding="utf-8")

    monkeypatch.setattr(sys, "executable", str(dist_dir / "sfmapi-colmap-api.exe"))
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    assert api_launcher.bundled_colmap_executable() == bundled


def test_native_launcher_reload_uses_import_string(monkeypatch):
    calls = []
    fake_uvicorn = type(
        "FakeUvicorn",
        (),
        {"run": staticmethod(lambda app, **kwargs: calls.append((app, kwargs)))},
    )
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setenv("SFMAPI_BACKEND", "colmap_cpp_inmemory")

    api_launcher.main_native(["--reload", "--mcp", "local"])

    assert calls == [
        (
            "scenemap.colmap.native.server:app",
            {
                "host": "127.0.0.1",
                "port": 8000,
                "log_level": "info",
                "reload": True,
            },
        )
    ]
    assert os.environ["SFMAPI_MCP_MODE"] == "local"


# ---- pycolmap launcher (ex sfmapi-pycolmap) ------------------------------


def test_pycolmap_launcher_configures_pycolmap_backend_and_mcp(monkeypatch, tmp_path: Path) -> None:
    fake_exe = _fake_colmap(tmp_path / "COLMAP" / "bin" / _exe_name())
    for key in (
        "SFMAPI_BACKEND",
        "SFMAPI_COLMAP_EXECUTABLE",
        "SFMAPI_EPHEMERAL",
        "SFMAPI_MCP_MODE",
        "SFMAPI_MCP_MOUNT_PATH",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PATH", "")

    args = api_launcher.build_parser("pycolmap").parse_args(
        [
            "--backend",
            "colmap_pycolmap",
            "--colmap-executable",
            str(fake_exe),
            "--mcp",
            "local",
            "--mcp-mount-path",
            "/agent",
        ]
    )
    selected = api_launcher.configure_environment(args)

    assert selected == fake_exe.resolve()
    assert os.environ["SFMAPI_BACKEND"] == "colmap_pycolmap"
    assert os.environ["SFMAPI_COLMAP_EXECUTABLE"] == str(fake_exe.resolve())
    assert os.environ["SFMAPI_EPHEMERAL"] == "true"
    assert os.environ["SFMAPI_MCP_MODE"] == "local"
    assert os.environ["SFMAPI_MCP_MOUNT_PATH"] == "/agent"


# ---- cli launcher (ex sfmapi-colmap-cli) ---------------------------------


def test_cli_launcher_configures_cli_backend_from_colmap_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_exe = _fake_colmap(tmp_path / "COLMAP" / "bin" / _exe_name())
    monkeypatch.delenv("SFMAPI_BACKEND", raising=False)
    monkeypatch.delenv("SFMAPI_COLMAP_EXECUTABLE", raising=False)
    monkeypatch.setenv("PATH", "")

    args = api_launcher.build_parser("cli").parse_args(
        [
            "--colmap-executable",
            str(fake_exe),
            "--mcp",
            "local",
            "--mcp-mount-path",
            "/agent",
        ]
    )
    selected = api_launcher.configure_environment(args)

    assert selected == fake_exe.resolve()
    assert os.environ["SFMAPI_BACKEND"] == "colmap_cli"
    assert os.environ["SFMAPI_COLMAP_EXECUTABLE"] == str(fake_exe.resolve())
    assert os.environ["SFMAPI_EPHEMERAL"] == "true"
    assert os.environ["SFMAPI_BLOB_BACKEND"] == "memory"
    assert os.environ["SFMAPI_MCP_MODE"] == "local"
    assert os.environ["SFMAPI_MCP_MOUNT_PATH"] == "/agent"
    assert str(fake_exe.parent.resolve()) in os.environ["PATH"].split(os.pathsep)


def test_cli_launcher_pins_colmap_cli_backend_despite_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The superseded sfmapi-colmap-cli-api always forced colmap_cli; the
    # unified launcher must preserve that (SFMAPI_BACKEND is not a
    # --backend default for the cli provider).
    monkeypatch.setenv("SFMAPI_BACKEND", "colmap_pycolmap")
    monkeypatch.delenv("SFMAPI_COLMAP_EXECUTABLE", raising=False)

    args = api_launcher.build_parser("cli").parse_args([])
    api_launcher.configure_environment(args)

    assert os.environ["SFMAPI_BACKEND"] == "colmap_cli"


def test_colmap_environment_accepts_install_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_dir = tmp_path / "colmap-install"
    fake_exe = _fake_colmap(install_dir / "bin" / _exe_name())
    monkeypatch.delenv("SFMAPI_COLMAP_EXECUTABLE", raising=False)
    monkeypatch.setenv("PATH", "")

    selected = configure_colmap_environment(install_dir, validate=True)

    assert selected == fake_exe.resolve()
    assert ColmapCliBackend().runtime_versions()["colmap_executable"] == str(fake_exe.resolve())


def test_invalid_configured_colmap_path_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="COLMAP executable not found"):
        configure_colmap_environment(tmp_path / "missing" / _exe_name(), validate=True)


def test_resolve_colmap_executable_accepts_direct_file(tmp_path: Path) -> None:
    fake_exe = _fake_colmap(tmp_path / _exe_name())

    assert resolve_colmap_executable(fake_exe) == fake_exe.resolve()
