from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

from sceneapi_map.instantsfm.backend import InstantSfMBackend


def _fake_instantsfm(root: Path) -> Path:
    (root / "instantsfm").mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname='instantsfm'\n", encoding="utf-8")
    return root


def test_action_catalog_exposes_instantsfm_actions(tmp_path: Path) -> None:
    backend = InstantSfMBackend(_fake_instantsfm(tmp_path / "InstantSfM"))

    actions = backend.list_backend_actions(include_schemas=True)
    action_ids = {action["action_id"] for action in actions}

    assert "instantsfm.extractFeatures" in action_ids
    assert "instantsfm.runGlobalSfm" in action_ids
    assert "instantsfm.runPipeline" in action_ids
    assert "instantsfm.runModule" in action_ids
    # The fake checkout resolves, so the backend advertises its one
    # portable capability (map.global, via the run_mapping adapter)
    # alongside the action catalog.
    assert backend.capabilities() == {"map.global"}
    extract = next(
        action for action in actions if action["action_id"] == "instantsfm.extractFeatures"
    )
    assert extract["input_schema"]["properties"]["feature_handler"]["enum"]
    # 3DGS training is categorised as a radiance-field tool, not dense MVS.
    gs = next(
        action for action in actions if action["action_id"] == "instantsfm.trainGaussianSplatting"
    )
    assert gs["category"] == "radiance_field"


def test_runtime_versions_reports_torch_cuda(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_name(index: int) -> str:
            assert index == 0
            return "Test GPU"

    torch = types.SimpleNamespace(
        __version__="2.11.0+cu128",
        version=types.SimpleNamespace(cuda="12.8"),
        cuda=_Cuda(),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    versions = InstantSfMBackend(_fake_instantsfm(tmp_path / "InstantSfM")).runtime_versions()

    assert versions["torch_status"] == "available"
    assert versions["torch_version"] == "2.11.0+cu128"
    assert versions["torch_cuda"] == "12.8"
    assert versions["torch_cuda_available"] is True
    assert versions["torch_cuda_device_count"] == 1
    assert versions["torch_cuda_device_name"] == "Test GPU"


def test_backend_contract_passes(tmp_path: Path) -> None:
    pytest.importorskip("sceneapi.backends")
    from sceneapi.backends import Backend, MappingBackend, SfmBackend, assert_backend_contract

    backend = InstantSfMBackend(_fake_instantsfm(tmp_path / "InstantSfM"))
    assert isinstance(backend, Backend)
    # InstantSfM satisfies the portable mapping protocol (run_mapping)
    # but not the whole SfmBackend surface (no feature / observation /
    # refinement / export / retrieval / localization methods).
    assert isinstance(backend, MappingBackend)
    assert not isinstance(backend, SfmBackend)
    assert_backend_contract(backend)


def test_validate_rejects_unknown_feature_handler(tmp_path: Path) -> None:
    backend = InstantSfMBackend(_fake_instantsfm(tmp_path / "InstantSfM"))

    result = backend.validate_backend_action(
        "instantsfm.extractFeatures",
        {"data_path": "dataset", "feature_handler": "unknown"},
    )

    assert result["valid"] is False
    assert "feature_handler must be one of" in result["errors"][0]["message"]


def test_run_extract_features_builds_module_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _fake_instantsfm(tmp_path / "InstantSfM")
    backend = InstantSfMBackend(root, python_executable="python")
    captured: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = backend.run_backend_action(
        "instantsfm.extractFeatures",
        {
            "data_path": "C:/data/project",
            "feature_handler": "colmap",
            "manual_config_name": "colmap",
            "single_camera": True,
        },
    )

    assert result["returncode"] == 0
    assert captured["args"] == [
        "python",
        "-m",
        "instantsfm.scripts.feat",
        "--data_path",
        "C:/data/project",
        "--manual_config_name",
        "colmap",
        "--feature_handler",
        "colmap",
        "--single_camera",
    ]


def test_run_mapping_global_stages_and_reads_back(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _fake_instantsfm(tmp_path / "InstantSfM")
    backend = InstantSfMBackend(root, python_executable="python")

    db_path = tmp_path / "feature.db"
    db_path.write_text("sqlite", encoding="utf-8")
    image_root = tmp_path / "imgs"
    image_root.mkdir()
    (image_root / "a.jpg").write_text("", encoding="utf-8")
    sparse_root = tmp_path / "out_sparse"
    job_dir = tmp_path / "job"

    captured: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        # InstantSfM scripts.sfm invocation -- emulate it writing a model
        # into <data_path>/sparse/0.
        if len(args) >= 3 and args[2] == "instantsfm.scripts.sfm":
            captured["args"] = args
            data_path = Path(args[args.index("--data_path") + 1])
            model = data_path / "sparse" / "0"
            model.mkdir(parents=True, exist_ok=True)
            for name in ("cameras.bin", "images.bin", "points3D.bin"):
                (model / name).write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="done", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    summaries, reconstructions = backend.run_mapping(
        kind="global",
        db_path=db_path,
        image_root=image_root,
        sparse_root=sparse_root,
        job_dir=job_dir,
        spec={"export_txt": True},
    )

    assert reconstructions == []
    assert len(summaries) == 1
    # The model was moved out of the staging dir into sparse_root/0.
    assert summaries[0]["model_path"] == str(sparse_root / "0")
    assert (sparse_root / "0" / "cameras.bin").exists()
    assert summaries[0]["engine"] == "instantsfm scripts.sfm"
    args = captured["args"]
    assert "--data_path" in args
    assert "--export_txt" in args
    # The staging directory is cleaned up after the model is recovered.
    assert not (job_dir / "instantsfm_stage").exists()


def test_run_mapping_rejects_non_global_kind(tmp_path: Path) -> None:
    pytest.importorskip("sceneapi.errors")
    from sceneapi.errors import CapabilityUnavailableError

    backend = InstantSfMBackend(_fake_instantsfm(tmp_path / "InstantSfM"))
    with pytest.raises(CapabilityUnavailableError):
        backend.run_mapping(
            kind="incremental",
            db_path=tmp_path / "db",
            image_root=tmp_path / "imgs",
            sparse_root=tmp_path / "sparse",
            job_dir=tmp_path / "job",
            spec={},
        )


def test_run_pipeline_executes_ordered_steps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _fake_instantsfm(tmp_path / "InstantSfM")
    backend = InstantSfMBackend(root, python_executable="python")
    modules: list[str] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        modules.append(args[2])
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = backend.run_backend_action(
        "instantsfm.runPipeline",
        {"data_path": str(tmp_path / "dataset"), "export_txt": True},
    )

    assert modules == ["instantsfm.scripts.feat", "instantsfm.scripts.sfm"]
    assert len(result["steps"]) == 2
