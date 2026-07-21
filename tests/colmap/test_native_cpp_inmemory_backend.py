from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from scenemap.colmap.native.backend import CapabilityUnavailableError, ValidationError
from scenemap.colmap.native.cpp_inmemory_backend import CppInmemoryBackend


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


def _write_pgm(path: Path, *, offset: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = bytes((index + offset) % 256 for index in range(16 * 16))
    path.write_bytes(b"P5\n16 16\n255\n" + pixels)


def test_capabilities_are_empty_when_cpp_extension_is_missing(monkeypatch: pytest.MonkeyPatch):
    backend = CppInmemoryBackend()

    def missing(name: str):
        if name == "sfmapi_colmap._cpp_inmemory":
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


@needs_cpp_extension
def test_cpp_inmemory_extracts_features_without_materializing_database(tmp_path: Path):
    images = tmp_path / "images"
    _write_pgm(images / "a.pgm")
    _write_pgm(images / "nested" / "b.pgm", offset=7)
    backend = CppInmemoryBackend()
    database_path = tmp_path / "database.db"

    result = backend.extract_features(
        database_path=database_path,
        image_root=images,
        image_list=["a.pgm", "nested/b.pgm"],
        options={"SiftExtraction.max_num_features": 5, "CppInmemory.descriptor_dim": 16},
    )

    assert "pairs.exhaustive" in backend.capabilities()
    assert result["engine"] == "sfmapi_colmap._cpp_inmemory.extract_features"
    assert result["in_memory"] is True
    assert result["num_images"] == 2
    assert result["descriptor_dim"] == 16
    assert result["num_keypoints"] == 10
    assert not database_path.exists()

    keypoints, descriptor_bytes, descriptor_dim = backend.read_keypoints(
        database_path=database_path,
        image_id=1,
    )
    assert len(keypoints) == 5
    assert len(descriptor_bytes) == 5 * 16
    assert descriptor_dim == 16


@needs_cpp_extension
def test_cpp_inmemory_matches_and_verifies_feature_pairs(tmp_path: Path):
    images = tmp_path / "images"
    for index, name in enumerate(["a.pgm", "b.pgm", "c.pgm"]):
        _write_pgm(images / name, offset=index * 11)
    backend = CppInmemoryBackend()
    database_path = tmp_path / "feature-store"
    backend.extract_features(
        database_path=database_path,
        image_root=images,
        image_list=["a.pgm", "b.pgm", "c.pgm"],
        options={"max_num_features": 4, "descriptor_dim": 8},
    )

    exhaustive = backend.match(
        database_path=database_path,
        mode="exhaustive",
        options={"cross_check": False, "max_distance": 999999999},
    )

    assert exhaustive["strategy"] == "exhaustive"
    assert exhaustive["num_pairs"] == 3
    assert exhaustive["num_matches"] == 12
    assert len(list(backend.iter_correspondences(database_path=database_path))) == 3

    verified = backend.verify_matches(database_path=database_path, options={})

    assert verified["num_verified_pairs"] == 3
    assert verified["num_verified_matches"] == 12
    geometries = list(backend.iter_two_view_geometries(database_path=database_path))
    assert len(geometries) == 3
    assert geometries[0][2].num_inliers == 4


@needs_cpp_extension
def test_cpp_inmemory_sequential_matching_filters_adjacent_pairs(tmp_path: Path):
    images = tmp_path / "images"
    for index, name in enumerate(["a.pgm", "b.pgm", "c.pgm"]):
        _write_pgm(images / name, offset=index)
    backend = CppInmemoryBackend()
    database_path = tmp_path / "feature-store"
    backend.extract_features(
        database_path=database_path,
        image_root=images,
        image_list=["a.pgm", "b.pgm", "c.pgm"],
        options={"max_num_features": 3},
    )

    result = backend.match(
        database_path=database_path,
        mode="sequential",
        options={"cross_check": False, "max_distance": 999999999},
    )

    assert result["num_pairs"] == 2
    assert result["num_matches"] == 6


@needs_cpp_extension
def test_cpp_inmemory_explicit_matching_filters_requested_pairs(tmp_path: Path):
    images = tmp_path / "images"
    for index, name in enumerate(["a.pgm", "b.pgm", "c.pgm"]):
        _write_pgm(images / name, offset=index)
    backend = CppInmemoryBackend()
    database_path = tmp_path / "feature-store"
    backend.extract_features(
        database_path=database_path,
        image_root=images,
        image_list=["a.pgm", "b.pgm", "c.pgm"],
        options={"max_num_features": 3},
    )

    result = backend.match(
        database_path=database_path,
        mode="explicit",
        options={
            "pairs": {"image_pairs": [{"image_name1": "a.pgm", "image_name2": "c.pgm"}]},
            "cross_check": False,
            "max_distance": 999999999,
        },
    )

    assert result["num_pairs"] == 1
    pairs = list(backend.iter_correspondences(database_path=database_path))
    assert [(pairs[0][0], pairs[0][1])] == [(1, 3)]


@needs_cpp_extension
def test_cpp_inmemory_rejects_unsupported_or_unprepared_operations(tmp_path: Path):
    images = tmp_path / "images"
    _write_pgm(images / "a.pgm")
    backend = CppInmemoryBackend()

    with pytest.raises(ValidationError):
        backend.match(database_path=tmp_path / "missing-store", mode="exhaustive", options={})
    with pytest.raises(CapabilityUnavailableError):
        backend.match(database_path=tmp_path / "missing-store", mode="spatial", options={})
    with pytest.raises(CapabilityUnavailableError):
        backend.run_mapping(
            kind="incremental",
            db_path=tmp_path / "store",
            image_root=images,
            sparse_root=tmp_path / "sparse",
            job_dir=tmp_path / "job",
            spec={},
        )
    with pytest.raises(CapabilityUnavailableError):
        backend.run_colmap_command("feature_extractor")
    with pytest.raises(CapabilityUnavailableError):
        backend.colmap_command_schema("feature_extractor")

    assert backend.list_colmap_commands() == []
    assert backend.list_colmap_command_schemas() == {}


@needs_cpp_extension
@pytest.mark.e2e
@pytest.mark.integration
def test_sfmapi_oneshot_features_uses_cpp_inmemory_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("sceneapi")
    pytest.importorskip("fastapi")
    monkeypatch.setenv("SFMAPI_MCP_MODE", "off")
    monkeypatch.setenv("SFMAPI_MCP_ENABLED", "false")
    from fastapi.testclient import TestClient
    from sceneapi.runtime import create_app, register_backend
    from sceneapi.testing import reset_runtime_for_tests_sync

    workspace = tmp_path / "workspace"
    monkeypatch.setenv("SFMAPI_BACKEND", "colmap_cpp_inmemory")
    reset_runtime_for_tests_sync(
        ephemeral=True,
        # Merge adaptation: the unified package installs all three
        # entry points, so lifespan autoload would re-register the real
        # colmap_cli/colmap_pycolmap providers over this test's fake.
        # Pin autoload off (the core's documented test convention).
        auto_load_backend_plugins=False,
        db_url="sqlite+aiosqlite:///file::memory:?cache=shared&uri=true",
        blob_backend="memory",
        queue_backend="inline",
        inline_tasks=True,
        workspace_root=workspace,
        blob_root=workspace / "_blobs",
        s3_cache_root=workspace / "_cache" / "s3",
    )
    register_backend("colmap_cpp_inmemory", CppInmemoryBackend)

    app = create_app()
    image_bytes = b"P5\n16 16\n255\n" + bytes(range(16 * 16))

    with TestClient(app) as client:
        capabilities = client.get("/v1/capabilities")
        assert capabilities.status_code == 200
        features = capabilities.json()["features"]
        assert capabilities.json()["backend"]["name"] == "colmap_cpp_inmemory"
        assert features["pairs.exhaustive"] is True
        assert features["pairs.explicit"] is True
        assert features["features.extract.sift"] is False

        response = client.post(
            "/v1/oneshot/features?max_num_features=4&use_gpu=false",
            content=image_bytes,
            headers={"Content-Type": "application/octet-stream"},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["kind"] == "oneshot.features"
    assert payload["runtime"]["backend"] == "colmap_cpp_inmemory"
    assert payload["image"]["byte_size"] == len(image_bytes)
    assert payload["features"]["count"] == 4
    assert payload["features"]["descriptor_dim"] == 32
    assert len(payload["features"]["keypoints"]) == 4
    assert payload["features"]["descriptors_b64"]
