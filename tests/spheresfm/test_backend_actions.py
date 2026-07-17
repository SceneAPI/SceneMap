from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sceneapi_map.spheresfm.backend import (
    SPHERESFM_CAPABILITIES,
    SPHERESFM_COMMANDS,
    SphereSfMBackend,
)


def _fake_colmap(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def test_action_catalog_exposes_spheresfm_actions(tmp_path: Path) -> None:
    backend = SphereSfMBackend(_fake_colmap(tmp_path / "colmap.exe"))

    actions = backend.list_backend_actions(include_schemas=True)
    action_ids = {action["action_id"] for action in actions}

    assert "spheresfm.reconstructPanoramaFolder" in action_ids
    assert "spheresfm.convertToCubemap" in action_ids
    assert "spheresfm.colmap.sphere_cubic_reprojecer" in action_ids
    assert "spheresfm.colmap.gui" not in action_ids
    assert len([action for action in action_ids if action.startswith("spheresfm.colmap.")]) == len(
        SPHERESFM_COMMANDS
    )
    # SphereSfM backs the full COLMAP-fork sparse pipeline as portable
    # capabilities with real wrapper methods; the action catalog
    # continues to expose every COLMAP-fork verb separately.
    assert backend.capabilities() == set(SPHERESFM_CAPABILITIES)


def test_capabilities_empty_without_executable(tmp_path: Path) -> None:
    # A deployment that cannot resolve the SphereSfM executable must not
    # advertise capabilities it cannot actually run.
    backend = SphereSfMBackend(tmp_path / "missing-colmap.exe")
    assert backend.capabilities() == set()


def test_backend_contract_passes(tmp_path: Path) -> None:
    pytest.importorskip("sceneapi.backends")
    from sceneapi.backends import (
        Backend,
        MappingBackend,
        SfmBackend,
        assert_backend_contract,
        has_backend_method,
    )

    backend = SphereSfMBackend(_fake_colmap(tmp_path / "colmap.exe"))
    assert isinstance(backend, Backend)
    # SphereSfM fully satisfies the portable mapping protocol (run_mapping).
    assert isinstance(backend, MappingBackend)
    # It implements the feature/match and spherical stage methods the
    # manifest advertises, but not the *whole* FeatureBackend protocol
    # (no standalone verify_matches — SphereSfM verifies inline) nor the
    # whole SphericalBackend protocol, so check method presence directly.
    assert has_backend_method(backend, "extract_features")
    assert has_backend_method(backend, "match")
    assert has_backend_method(backend, "convert_spherical_to_cubemap")
    assert not has_backend_method(backend, "verify_matches")
    # SphereSfM is still not a full SfmBackend (no observation readers,
    # refinement, export, retrieval, or localization surface).
    assert not isinstance(backend, SfmBackend)
    assert_backend_contract(backend)


def test_validate_rejects_bad_matching_mode(tmp_path: Path) -> None:
    backend = SphereSfMBackend(_fake_colmap(tmp_path / "colmap.exe"))

    result = backend.validate_backend_action(
        "spheresfm.reconstructPanoramaFolder",
        {
            "image_path": "images",
            "workspace_path": "workspace",
            "matching_mode": "bad",
        },
    )

    assert result["valid"] is False
    assert "matching_mode must be one of" in result["errors"][0]["message"]


def test_reconstruct_panorama_folder_builds_spherical_sequence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    exe = _fake_colmap(tmp_path / "build" / "colmap.exe")
    image_path = tmp_path / "images"
    image_path.mkdir()
    backend = SphereSfMBackend(exe)
    captured: list[list[str]] = []
    action = backend.get_backend_action("spheresfm.reconstructPanoramaFolder")
    properties = action["input_schema"]["properties"]
    assert properties["use_gpu"]["default"] is True
    assert properties["max_num_features"]["default"] == 8192

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.append(args)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = backend.run_backend_action(
        "spheresfm.reconstructPanoramaFolder",
        {
            "image_path": str(image_path),
            "workspace_path": str(tmp_path / "workspace"),
            "camera_params": "1,3520,1760",
            "matching_mode": "spatial",
            "use_gpu": False,
            "max_num_features": 2048,
        },
    )

    commands = [args[1] for args in captured]
    assert commands == ["database_creator", "feature_extractor", "spatial_matcher", "mapper"]
    feature_args = captured[1]
    assert "--ImageReader.camera_model" in feature_args
    assert "SPHERE" in feature_args
    assert "--SiftExtraction.use_gpu" in feature_args
    assert "0" in feature_args
    assert "--SiftExtraction.max_num_features" in feature_args
    assert "2048" in feature_args
    matcher_args = captured[2]
    assert "--SiftMatching.use_gpu" in matcher_args
    assert "0" in matcher_args
    mapper_args = captured[3]
    assert "--Mapper.sphere_camera" in mapper_args
    assert "1" in mapper_args
    assert result["database_path"].endswith("database.db")


def test_generic_command_builds_colmap_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    exe = _fake_colmap(tmp_path / "colmap.exe")
    backend = SphereSfMBackend(exe)
    captured: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = backend.run_backend_action(
        "spheresfm.colmap.feature_extractor",
        {
            "options": {
                "database_path": "db.sqlite",
                "image_path": "images",
                "ImageReader.camera_model": "SPHERE",
            }
        },
    )

    assert result["returncode"] == 0
    assert captured["args"] == [
        str(exe.resolve()),
        "feature_extractor",
        "--database_path",
        "db.sqlite",
        "--image_path",
        "images",
        "--ImageReader.camera_model",
        "SPHERE",
    ]
