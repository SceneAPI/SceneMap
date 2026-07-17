from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from sceneapi_map.colmap.native.backend import (
    COLMAP_COMMANDS,
    CapabilityUnavailableError,
    ValidationError,
)
from sceneapi_map.colmap.native.cpp_native_backend import CppNativeBackend

OPTIONAL_DISABLED_COMMANDS = {"delaunay_mesher": "requires CGAL"}


def _cpp_extension_available() -> bool:
    try:
        importlib.import_module("sfmapi_colmap._cpp_inmemory")
    except (ImportError, RuntimeError):
        return False
    return True


# Merge adaptation: the unified package does not build the pybind11 demo
# extension (the superseded sfmapi_colmap repo did, via scikit-build-core
# — see README). Tests that drive the real extension skip when it is not
# installed; the CapabilityUnavailable and fake-extension tests still run.
needs_cpp_extension = pytest.mark.skipif(
    not _cpp_extension_available(),
    reason=(
        "sfmapi_colmap._cpp_inmemory extension not installed; build/install it "
        "from the superseded sfmapi_colmap repo"
    ),
)


class _FakeCppNative:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[str]]] = []

    def version(self) -> str:
        return "fake-cpp"

    def run_command(self, argv: list[str], path_prefix: list[str]) -> dict:
        self.calls.append((argv, path_prefix))
        return {
            "returncode": 0,
            "stdout": f"ran {' '.join(argv)}",
            "stderr": "",
        }


def test_cpp_native_capabilities_are_empty_when_extension_is_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    backend = CppNativeBackend()

    def missing(name: str):
        if name == "sfmapi_colmap._cpp_inmemory":
            raise ModuleNotFoundError(name)
        return importlib.import_module(name)

    monkeypatch.setattr(importlib, "import_module", missing)

    assert backend.capabilities() == set()
    with pytest.raises(CapabilityUnavailableError):
        backend.run_colmap_command("version")


def test_cpp_native_backend_uses_cpp_extension_for_generic_colmap_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fake_colmap = tmp_path / "colmap"
    fake_colmap.write_text("", encoding="utf-8")
    fake_cpp = _FakeCppNative()
    backend = CppNativeBackend(executable=fake_colmap)
    monkeypatch.setattr(backend, "_require_cpp", lambda capability: fake_cpp)

    result = backend.run_colmap_command("version")

    assert result["command"] == "version"
    assert result["returncode"] == 0
    assert result["stdout"].startswith("ran ")
    assert fake_cpp.calls[0][0] == [str(fake_colmap), "version"]


def test_cpp_native_backend_exposes_colmap_surface_as_backend_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fake_colmap = tmp_path / "colmap"
    fake_colmap.write_text("", encoding="utf-8")
    backend = CppNativeBackend(executable=fake_colmap)
    monkeypatch.setattr(backend, "_require_cpp", lambda capability: _FakeCppNative())

    capabilities = backend.capabilities()
    actions = backend.list_backend_actions()
    action_ids = {action["action_id"] for action in actions}

    assert "map.global" in capabilities
    assert "map.incremental" in capabilities
    assert "export.nvm" in capabilities
    assert not any(capability.startswith("colmap.") for capability in capabilities)
    assert "colmap.feature_extractor" in action_ids
    assert all(f"colmap.{command}" in action_ids for command in COLMAP_COMMANDS)
    assert backend.list_colmap_commands() == list(COLMAP_COMMANDS)
    assert "version" in backend.list_colmap_commands()
    assert "gui" not in backend.list_colmap_commands()


def test_cpp_native_backend_validates_generic_command_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fake_colmap = tmp_path / "colmap"
    fake_colmap.write_text("", encoding="utf-8")
    fake_cpp = _FakeCppNative()
    backend = CppNativeBackend(executable=fake_colmap)
    monkeypatch.setattr(backend, "_require_cpp", lambda capability: fake_cpp)
    monkeypatch.setattr(
        backend,
        "colmap_command_schema",
        lambda command: {
            "command": command,
            "available": True,
            "options": [
                {
                    "name": "input_path",
                    "flags": ["--input_path"],
                    "takes_value": True,
                    "type": "string",
                    "choices": [],
                    "schema": {"type": "string", "format": "path"},
                },
                {
                    "name": "input_type",
                    "flags": ["--input_type"],
                    "takes_value": True,
                    "type": "string",
                    "choices": ["dense", "sparse"],
                    "schema": {"type": "string", "enum": ["dense", "sparse"]},
                },
            ],
        },
    )

    result = backend.run_colmap_command(
        "delaunay_mesher",
        options={"input_path": tmp_path / "dense", "input_type": "dense"},
    )

    assert result["returncode"] == 0
    assert fake_cpp.calls[0][0] == [
        str(fake_colmap),
        "delaunay_mesher",
        "--input_path",
        str(tmp_path / "dense"),
        "--input_type",
        "dense",
    ]
    validation = backend.validate_backend_action(
        "colmap.delaunay_mesher",
        {"input_path": tmp_path / "dense", "input_type": "dense"},
    )
    assert validation["valid"] is True
    assert validation["normalized_inputs"]["input_type"] == "dense"

    result = backend.run_backend_action(
        "colmap.delaunay_mesher",
        {"input_path": tmp_path / "dense", "input_type": "dense"},
    )
    assert result["returncode"] == 0
    with pytest.raises(ValidationError, match="must be one of"):
        backend.run_colmap_command("delaunay_mesher", options={"input_type": "bad"})
    invalid = backend.validate_backend_action("colmap.delaunay_mesher", {"input_type": "bad"})
    assert invalid["valid"] is False


@needs_cpp_extension
@pytest.mark.integration
@pytest.mark.needs_colmap
def test_cpp_native_backend_runs_real_colmap_version_command(colmap_executable: Path):
    backend = CppNativeBackend(executable=colmap_executable)

    result = backend.run_colmap_command("version")

    assert result["returncode"] == 0
    assert "COLMAP" in result["stdout"]


@needs_cpp_extension
@pytest.mark.integration
@pytest.mark.needs_colmap
@pytest.mark.parametrize("command", COLMAP_COMMANDS)
def test_cpp_native_bridge_can_reach_every_non_gui_colmap_command(
    colmap_executable: Path,
    command: str,
):
    backend = CppNativeBackend(executable=colmap_executable)
    schema = backend.colmap_command_schema(command)
    args = [str(colmap_executable), command]
    if command != "version":
        args.append("-h")

    try:
        result = backend._run(args)
    except ValidationError as exc:
        optional_reason = OPTIONAL_DISABLED_COMMANDS.get(command)
        if optional_reason and optional_reason in str(exc):
            assert schema["available"] is False
            assert schema["schema_source"] == "colmap_source_fallback"
            assert schema["options"]
            return
        raise

    text = result.stdout + result.stderr
    assert "COLMAP" in text
    assert schema["available"] is True
    assert schema["command"] == command
    assert schema["option_count"] == len(schema["options"])
    assert len({option["name"] for option in schema["options"]}) == len(schema["options"])
    if command == "version":
        assert "Commit" in text
        assert schema["options"] == []
    elif command == "help":
        assert "Available commands" in text
        assert schema["options"] == []
    else:
        assert "-h [ --help ]" in text
        assert schema["options"]
        for option in schema["options"]:
            assert option["flags"]
            assert "schema" in option
            assert "type" in option["schema"]
