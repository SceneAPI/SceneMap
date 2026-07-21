from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scenemap.colmap.native.backend import ColmapCliBackend as NativeColmapCliBackend
from scenemap.colmap.pycolmap.backend import ColmapCliBackend as PycolmapColmapCliBackend

# sfmapi_colmap and sfmapi_pycolmap carried this test as import-only
# variants over their own (distinct) ColmapCliBackend implementations;
# parametrize to keep both implementations covered. The colmap_cli repo
# never shipped this suite, so its backend is not added here.
PROVIDER_BACKENDS = [
    pytest.param(NativeColmapCliBackend, id="native"),
    pytest.param(PycolmapColmapCliBackend, id="pycolmap"),
]


def _count_rows(database_path: Path, table: str) -> int:
    with sqlite3.connect(database_path) as conn:
        return int(conn.execute(f"select count(*) from {table}").fetchone()[0])


def _count_positive_rows(database_path: Path, table: str) -> int:
    with sqlite3.connect(database_path) as conn:
        return int(conn.execute(f"select count(*) from {table} where rows > 0").fetchone()[0])


@pytest.mark.integration
@pytest.mark.needs_colmap
@pytest.mark.needs_sample_data
@pytest.mark.slow
@pytest.mark.parametrize("backend_cls", PROVIDER_BACKENDS)
def test_real_colmap_sparse_pipeline_on_official_sample_subset(
    tmp_path: Path,
    colmap_executable: Path,
    colmap_sample_subset,
    backend_cls,
):
    backend = backend_cls(executable=colmap_executable)
    database_path = tmp_path / "database.db"

    feature_result = backend.extract_features(
        database_path=database_path,
        image_root=colmap_sample_subset.image_root,
        image_list=colmap_sample_subset.image_names,
        options={
            "ImageReader.single_camera": True,
            "FeatureExtraction.use_gpu": False,
            "max_num_features": 2048,
        },
    )

    assert feature_result["num_images"] == len(colmap_sample_subset.image_names)
    assert _count_rows(database_path, "images") == len(colmap_sample_subset.image_names)
    assert _count_positive_rows(database_path, "keypoints") >= 2
    assert _count_positive_rows(database_path, "descriptors") >= 2

    match_result = backend.match(
        database_path=database_path,
        mode="exhaustive",
        options={"FeatureMatching.use_gpu": False},
    )

    assert match_result["engine"] == "colmap exhaustive_matcher"
    assert _count_positive_rows(database_path, "matches") >= 1
    assert list(backend.iter_correspondences(database_path=database_path))

    verify_result = backend.verify_matches(
        database_path=database_path,
        options={"max_error": 4.0},
    )

    assert verify_result["num_verified_pairs"] >= 1
    assert _count_positive_rows(database_path, "two_view_geometries") >= 1

    summaries, reconstructions = backend.run_mapping(
        kind="incremental",
        db_path=database_path,
        image_root=colmap_sample_subset.image_root,
        sparse_root=tmp_path / "sparse",
        job_dir=tmp_path / "job",
        spec={"mapper": {"min_num_matches": 10}},
    )

    assert summaries
    assert reconstructions
    assert max(item["num_reg_images"] for item in summaries) >= 2
    assert sum(item["num_points3D"] for item in summaries) >= 1

    primary_model_path = Path(summaries[0]["model_path"])
    txt_export = tmp_path / "exports" / "txt"
    ply_export = tmp_path / "exports" / "model.ply"
    nvm_export = tmp_path / "exports" / "model.nvm"

    backend.export(model_path=primary_model_path, output_path=txt_export, format="colmap_text")
    backend.export(model_path=primary_model_path, output_path=ply_export, format="ply")
    backend.export(model_path=primary_model_path, output_path=nvm_export, format="nvm")

    assert (txt_export / "cameras.txt").is_file()
    assert (txt_export / "images.txt").is_file()
    assert (txt_export / "points3D.txt").is_file()
    assert ply_export.read_text(encoding="utf-8", errors="ignore").startswith("ply")
    assert nvm_export.read_text(encoding="utf-8", errors="ignore").startswith("NVM")

    analyzer = backend.run_colmap_command(
        "model_analyzer",
        options={"path": primary_model_path},
    )

    analyzer_text = analyzer["stdout"] + analyzer["stderr"]
    assert "Registered images" in analyzer_text
