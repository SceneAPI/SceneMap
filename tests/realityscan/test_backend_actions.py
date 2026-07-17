from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from sceneapi_map.realityscan.backend import REALITYSCAN_COMMANDS, RealityScanCliBackend


def _fake_realityscan(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def _exe_name() -> str:
    return "RealityScan.exe" if os.name == "nt" else "RealityScan"


def test_action_catalog_exposes_native_actions() -> None:
    backend = RealityScanCliBackend()
    actions = backend.list_backend_actions(include_schemas=True)
    action_ids = {action["action_id"] for action in actions}

    assert "realityscan.runSequence" in action_ids
    assert "realityscan.reconstructImageFolder" in action_ids
    assert "realityscan.align" in action_ids
    assert "realityscan.exportSelectedModel" in action_ids
    # Headline mesh / texture / ortho / point-cloud verbs are now in the
    # curated catalog — they are core photogrammetry workflow steps.
    assert "realityscan.calculateOrthoProjection" in action_ids
    assert "realityscan.exportOrthoProjection" in action_ids
    assert "realityscan.exportDEM" in action_ids
    assert "realityscan.calculateTexture" in action_ids
    assert "realityscan.unwrap" in action_ids
    assert "realityscan.calculateVertexColors" in action_ids
    assert "realityscan.simplify" in action_ids
    assert "realityscan.cleanModel" in action_ids
    assert "realityscan.exportPointCloud" in action_ids
    # Canned multi-step workflows live alongside the per-verb actions.
    assert "realityscan.reconstructToTexturedMesh" in action_ids
    assert "realityscan.reconstructToOrthophoto" in action_ids
    assert "realityscan.alignOnly" in action_ids
    # Genuine UI-only / delegation / upload / paid add-on verbs stay excluded.
    assert "realityscan.dtmClassify" not in action_ids
    assert "realityscan.uploadToSketchfab" not in action_ids
    assert "realityscan.delegateTo" not in action_ids
    assert len(
        [action for action in actions if action["action_id"].startswith("realityscan.")]
    ) >= len(REALITYSCAN_COMMANDS)
    assert backend.capabilities() == set()


def test_gpu_required_commands_are_all_reachable() -> None:
    from sceneapi_map.realityscan.backend import (
        _GPU_REQUIRED_COMMANDS,
        REALITYSCAN_COMMAND_SET,
    )

    # Regression: calculateTexture / calculateVertexColors / unwrap used to be
    # in the gpu_required set but missing from REALITYSCAN_COMMANDS, so they
    # were unreachable through the action catalog.
    assert _GPU_REQUIRED_COMMANDS
    assert _GPU_REQUIRED_COMMANDS <= REALITYSCAN_COMMAND_SET


def test_artifact_contracts_describe_alignment_exports() -> None:
    backend = RealityScanCliBackend()
    contracts = backend.list_backend_artifact_contracts()
    by_id = {contract["contract_id"]: contract for contract in contracts}

    assert set(by_id) == {"realityscan.registration", "realityscan.sparse_point_cloud"}
    for contract in by_id.values():
        assert contract["stage"] == "mapping"
        assert contract["capability"] is None
        assert contract["emits"] == ["reconstruction.sparse.v1"]


def test_textured_mesh_workflow_builds_full_sequence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_exe = _fake_realityscan(tmp_path / "RealityScan" / _exe_name())
    image_folder = tmp_path / "images"
    image_folder.mkdir()
    backend = RealityScanCliBackend(fake_exe)
    captured: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = backend.run_backend_action(
        "realityscan.reconstructToTexturedMesh",
        {
            "image_folder": str(image_folder),
            "project_path": str(tmp_path / "out" / "scan.rsproj"),
            "export_model_path": str(tmp_path / "out" / "model.obj"),
        },
    )

    args = captured["args"]
    assert isinstance(args, list)
    for verb in (
        "-align",
        "-calculateHighModel",
        "-unwrap",
        "-calculateTexture",
        "-exportSelectedModel",
    ):
        assert verb in args
    assert result["workflow"] == "reconstruct_to_textured_mesh"


def test_validate_sequence_rejects_unknown_command() -> None:
    backend = RealityScanCliBackend()

    result = backend.validate_backend_action(
        "realityscan.runSequence",
        {"commands": [{"name": "notACommand"}]},
    )

    assert result["valid"] is False
    assert "unknown RealityScan command" in result["errors"][0]["message"]


def test_run_command_builds_realityscan_cli_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_exe = _fake_realityscan(tmp_path / "RealityScan" / _exe_name())
    backend = RealityScanCliBackend(fake_exe)
    captured: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = backend.run_backend_action("realityscan.align", {"args": [], "append_quit": True})

    assert result["returncode"] == 0
    args = captured["args"]
    assert isinstance(args, list)
    assert args[:5] == [
        str(fake_exe.resolve()),
        "-headless",
        "-stdConsole",
        "-set",
        "appQuitOnError=true",
    ]
    assert "-align" in args
    assert args[-1] == "-quit"


def test_reconstruct_image_folder_builds_project_sequence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_exe = _fake_realityscan(tmp_path / "RealityScan" / _exe_name())
    image_folder = tmp_path / "images"
    image_folder.mkdir()
    project_path = tmp_path / "out" / "scan.rsproj"
    backend = RealityScanCliBackend(fake_exe)
    captured: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = backend.run_backend_action(
        "realityscan.reconstructImageFolder",
        {
            "image_folder": str(image_folder),
            "project_path": str(project_path),
            "quality": "preview",
        },
    )

    args = captured["args"]
    assert isinstance(args, list)
    assert "-addFolder" in args
    assert str(image_folder) in args
    assert "-align" in args
    assert "-calculatePreviewModel" in args
    assert "-save" in args
    assert result["project_path"] == str(project_path)
