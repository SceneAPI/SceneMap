from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sceneapi_map.realityscan.backend import REALITYSCAN_COMMANDS, RealityScanCliBackend

sfmapi_backends = pytest.importorskip("sceneapi.backends")


def _fake_realityscan(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def test_all_reality_cli_actions_validate_through_sfmapi_adapter() -> None:
    backend = RealityScanCliBackend()

    sfmapi_backends.assert_backend_contract(backend)
    actions = sfmapi_backends.list_backend_actions(backend, include_schemas=True)
    action_ids = {str(action["action_id"]) for action in actions}

    assert "realityscan.runSequence" in action_ids
    assert "realityscan.reconstructImageFolder" in action_ids
    # Mesh / texture / ortho verbs are core photogrammetry steps — now curated.
    assert "realityscan.calculateOrthoProjection" in action_ids
    assert "realityscan.calculateTexture" in action_ids
    # Genuine UI-only / upload / paid add-on verbs stay excluded.
    assert "realityscan.dtmClassify" not in action_ids
    assert "realityscan.uploadToSketchfab" not in action_ids
    for command in REALITYSCAN_COMMANDS:
        action_id = f"realityscan.{command}"
        assert action_id in action_ids
        validation = sfmapi_backends.validate_backend_action(
            action_id,
            {"args": [], "timeout_seconds": 1},
            backend,
        )
        assert validation["valid"], (action_id, validation)

    assert sfmapi_backends.list_backend_config_schemas(backend) == []


def test_artifact_contracts_validate_through_sfmapi_adapter() -> None:
    backend = RealityScanCliBackend()

    # Action-only artifact contracts: capability=None is allowed, and the
    # core adapter must report zero contract violations.
    assert sfmapi_backends.backend_artifact_contract_violations(backend) == []
    contracts = sfmapi_backends.list_backend_artifact_contracts(backend)
    contract_ids = {str(contract["contract_id"]) for contract in contracts}
    assert contract_ids == {"realityscan.registration", "realityscan.sparse_point_cloud"}
    for contract in contracts:
        assert contract["capability"] is None
        assert contract["stage"] == "mapping"
        assert "reconstruction.sparse.v1" in contract["emits"]


def test_run_sequence_executes_through_sfmapi_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_exe = _fake_realityscan(tmp_path / "RealityCapture_3.0" / "RealityCapture.exe")
    backend = RealityScanCliBackend(fake_exe)
    captured: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = sfmapi_backends.run_backend_action(
        "realityscan.runSequence",
        {"commands": [{"name": "quit"}], "append_quit": False},
        backend=backend,
        workspace=tmp_path,
    )

    assert result["returncode"] == 0
    assert result["stdout"] == "ok"
    assert captured["args"] == [
        str(fake_exe.resolve()),
        "-headless",
        "-stdConsole",
        "-set",
        "appQuitOnError=true",
        "-set",
        "suppressErrors=true",
        "-quit",
    ]


def test_reconstruct_image_folder_executes_through_sfmapi_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_exe = _fake_realityscan(tmp_path / "RealityCapture_3.0" / "RealityCapture.exe")
    image_folder = tmp_path / "images"
    image_folder.mkdir()
    project_stem = tmp_path / "out" / "scan"
    backend = RealityScanCliBackend(fake_exe)
    captured: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = sfmapi_backends.run_backend_action(
        "realityscan.reconstructImageFolder",
        {
            "image_folder": str(image_folder),
            "project_path": str(project_stem),
            "quality": "preview",
        },
        backend=backend,
        workspace=tmp_path,
    )

    args = captured["args"]
    assert isinstance(args, list)
    assert "-addFolder" in args
    assert str(image_folder) in args
    assert "-align" in args
    assert "-calculatePreviewModel" in args
    assert "-save" in args
    assert result["project_path"].endswith(".rcproj")
