from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path

import pytest

from scenemap.colmap.pycolmap.backend import CapabilityUnavailableError
from scenemap.colmap.pycolmap_backend import PycolmapBackend


def _count_positive_rows(database_path: Path, table: str) -> int:
    with sqlite3.connect(database_path) as conn:
        return int(conn.execute(f"select count(*) from {table} where rows > 0").fetchone()[0])


def test_capabilities_are_empty_when_pycolmap_is_missing(monkeypatch: pytest.MonkeyPatch):
    backend = PycolmapBackend()

    def missing(name: str):
        if name == "pycolmap":
            raise ModuleNotFoundError(name)
        return importlib.import_module(name)

    monkeypatch.setattr(importlib, "import_module", missing)

    assert backend.capabilities() == set()
    with pytest.raises(CapabilityUnavailableError):
        backend.extract_features(
            database_path=Path("database.db"),
            image_root=Path("images"),
            image_list=[],
            options={},
        )


@pytest.mark.needs_pycolmap
def test_pycolmap_option_translation_accepts_cli_style_names(monkeypatch: pytest.MonkeyPatch):
    pycolmap = pytest.importorskip("pycolmap")
    backend = PycolmapBackend()
    monkeypatch.setenv("SFMAPI_COLMAP_USE_GPU", "0")

    reader, extraction, camera_mode = backend._feature_options(
        pycolmap,
        {
            "ImageReader.single_camera": True,
            "ImageReader.camera_model": "SIMPLE_PINHOLE",
            "FeatureExtraction.use_gpu": False,
            "SiftExtraction.max_num_features": 1234,
        },
    )

    assert camera_mode == pycolmap.CameraMode.SINGLE
    assert reader.camera_model == "SIMPLE_PINHOLE"
    assert extraction.use_gpu is False
    assert extraction.sift.max_num_features == 1234

    matching, pairing, verification = backend._matching_options(
        pycolmap,
        "exhaustive_matcher",
        {
            "FeatureMatching.use_gpu": False,
            "SiftMatching.max_ratio": 0.7,
            "ExhaustiveMatching.block_size": 8,
            "TwoViewGeometry.max_error": 3.5,
        },
    )

    assert matching.use_gpu is False
    assert matching.sift.max_ratio == 0.7
    assert pairing.block_size == 8
    assert verification.ransac.max_error == 3.5


@pytest.mark.needs_pycolmap
def test_runtime_versions_report_pycolmap_and_colmap_41_source():
    pytest.importorskip("pycolmap")

    versions = PycolmapBackend().runtime_versions()

    assert versions["pycolmap"] != "missing"
    assert versions["colmap_source_version"] == "4.1.0.dev0"
    assert len(versions["colmap_source_sha"]) >= 7


@pytest.mark.integration
@pytest.mark.needs_pycolmap
@pytest.mark.needs_sample_data
@pytest.mark.slow
def test_real_pycolmap_sparse_pipeline_on_official_sample_subset(
    tmp_path: Path,
    colmap_sample_subset,
):
    pytest.importorskip("pycolmap")
    backend = PycolmapBackend()
    database_path = tmp_path / "database.db"

    feature_result = backend.extract_features(
        database_path=database_path,
        image_root=colmap_sample_subset.image_root,
        image_list=colmap_sample_subset.image_names,
        options={
            "ImageReader.single_camera": True,
            "FeatureExtraction.use_gpu": False,
            "SiftExtraction.max_num_features": 2048,
        },
    )

    assert feature_result["engine"] == "pycolmap.extract_features"
    assert _count_positive_rows(database_path, "keypoints") >= 2
    assert _count_positive_rows(database_path, "descriptors") >= 2

    match_result = backend.match(
        database_path=database_path,
        mode="exhaustive",
        options={"FeatureMatching.use_gpu": False},
    )

    assert match_result["engine"].startswith("pycolmap.")
    assert _count_positive_rows(database_path, "matches") >= 1

    verify_result = backend.verify_matches(
        database_path=database_path,
        options={"TwoViewGeometry.max_error": 4.0},
    )

    assert verify_result["num_verified_pairs"] >= 1

    summaries, reconstructions = backend.run_mapping(
        kind="incremental",
        db_path=database_path,
        image_root=colmap_sample_subset.image_root,
        sparse_root=tmp_path / "sparse",
        job_dir=tmp_path / "job",
        spec={"mapper": {"min_num_matches": 10, "min_model_size": 2}},
    )

    assert summaries
    assert reconstructions
    assert max(item["num_reg_images"] for item in summaries) >= 2

    primary_model_path = Path(summaries[0]["model_path"])
    txt_export = tmp_path / "exports" / "txt"
    ply_export = tmp_path / "exports" / "model.ply"

    backend.export(model_path=primary_model_path, output_path=txt_export, format="colmap_text")
    backend.export(model_path=primary_model_path, output_path=ply_export, format="ply")

    assert (txt_export / "cameras.txt").is_file()
    assert (txt_export / "images.txt").is_file()
    assert (txt_export / "points3D.txt").is_file()
    assert ply_export.read_text(encoding="utf-8", errors="ignore").startswith("ply")


@pytest.mark.integration
@pytest.mark.needs_pycolmap
@pytest.mark.needs_sample_data
@pytest.mark.slow
def test_real_pycolmap_localize_from_memory_on_official_sample_subset(
    tmp_path: Path,
    colmap_sample_subset,
):
    """End-to-end exercise of ``localize_from_memory`` against a real
    pycolmap reconstruction — closes the gap where only the
    CLI-backend "raises CapabilityUnavailable" wiring was covered and
    ``pycolmap.estimate_and_refine_absolute_pose`` itself never ran.

    Builds a sparse model from the sample subset, then localizes one of
    the source images back against that model — it is in the reference
    set, so the pose solve must succeed.
    """
    pytest.importorskip("pycolmap")
    backend = PycolmapBackend()
    database_path = tmp_path / "database.db"

    backend.extract_features(
        database_path=database_path,
        image_root=colmap_sample_subset.image_root,
        image_list=colmap_sample_subset.image_names,
        options={
            "ImageReader.single_camera": True,
            "FeatureExtraction.use_gpu": False,
            "SiftExtraction.max_num_features": 2048,
        },
    )
    backend.match(
        database_path=database_path,
        mode="exhaustive",
        options={"FeatureMatching.use_gpu": False},
    )
    backend.verify_matches(database_path=database_path, options={"TwoViewGeometry.max_error": 4.0})
    summaries, _ = backend.run_mapping(
        kind="incremental",
        db_path=database_path,
        image_root=colmap_sample_subset.image_root,
        sparse_root=tmp_path / "sparse",
        job_dir=tmp_path / "job",
        spec={"mapper": {"min_num_matches": 10, "min_model_size": 2}},
    )
    assert summaries

    query_name = colmap_sample_subset.image_names[0]
    query_image = colmap_sample_subset.image_root / query_name

    result = backend.localize_from_memory(
        sparse_dir=tmp_path / "sparse",
        query_image=query_image,
        spec={"database_path": str(database_path)},
    )

    assert result["engine"] == "pycolmap.estimate_and_refine_absolute_pose"
    # The query image built the model, so the absolute-pose solve must
    # land — a success result carries the recovered rotation/translation.
    assert result["success"] is True, result
    assert result["num_correspondences"] >= 4
    assert "rotation" in result
    assert "translation" in result
    assert len(result["translation"]) == 3
