from __future__ import annotations

from pathlib import Path

import pytest

from sceneapi_map.colmap.native.backend import ColmapCliBackend as NativeColmapCliBackend
from sceneapi_map.colmap.pycolmap.backend import ColmapCliBackend as PycolmapColmapCliBackend
from sceneapi_map.colmap.pycolmap_backend import PycolmapBackend

# sfmapi_colmap's suite was a superset of sfmapi_pycolmap's (one extra
# artifact-input test); the CLI-driven tests are parametrized over both
# repos' distinct ColmapCliBackend implementations, and the pycolmap
# pipeline test runs once against the reconciled PycolmapBackend.


def _cpu_cli_backend_cls(base: type) -> type:
    class CpuColmapCliBackend(base):
        def extract_features(
            self,
            *,
            database_path: Path,
            image_root: Path,
            image_list: list[str],
            options: dict,
        ) -> dict:
            merged = {
                "ImageReader.single_camera": True,
                "FeatureExtraction.use_gpu": False,
                "max_num_features": 2048,
                **options,
            }
            return super().extract_features(
                database_path=database_path,
                image_root=image_root,
                image_list=image_list,
                options=merged,
            )

        def match(self, *, database_path: Path, mode: str, options: dict) -> dict:
            merged = {**options, "use_gpu": False}
            return super().match(
                database_path=database_path,
                mode=mode,
                options=merged,
            )

    return CpuColmapCliBackend


CLI_BACKENDS = [
    pytest.param(NativeColmapCliBackend, id="native"),
    pytest.param(PycolmapColmapCliBackend, id="pycolmap"),
]


class CpuPycolmapBackend(PycolmapBackend):
    def extract_features(
        self,
        *,
        database_path: Path,
        image_root: Path,
        image_list: list[str],
        options: dict,
    ) -> dict:
        merged = {
            "ImageReader.single_camera": True,
            "FeatureExtraction.use_gpu": False,
            "SiftExtraction.max_num_features": 2048,
            **options,
        }
        return super().extract_features(
            database_path=database_path,
            image_root=image_root,
            image_list=image_list,
            options=merged,
        )

    def match(self, *, database_path: Path, mode: str, options: dict) -> dict:
        merged = {**options, "FeatureMatching.use_gpu": False}
        return super().match(
            database_path=database_path,
            mode=mode,
            options=merged,
        )

    def run_mapping(
        self,
        *,
        kind: str,
        db_path: Path,
        image_root: Path,
        sparse_root: Path,
        job_dir: Path,
        spec: dict,
        pose_priors: dict | None = None,
    ):
        merged = {
            "min_model_size": 2,
            "min_num_matches": 10,
            **spec,
        }
        return super().run_mapping(
            kind=kind,
            db_path=db_path,
            image_root=image_root,
            sparse_root=sparse_root,
            job_dir=job_dir,
            spec=merged,
            pose_priors=pose_priors,
        )


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.needs_colmap
@pytest.mark.needs_sample_data
@pytest.mark.slow
@pytest.mark.parametrize("cli_backend_cls", CLI_BACKENDS)
def test_sfmapi_http_pipeline_runs_colmap_backend_on_official_sample_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    colmap_executable: Path,
    colmap_sample_subset,
    cli_backend_cls,
):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from sceneapi.runtime import create_app, register_backend
    from sceneapi.testing import reset_runtime_for_tests_sync

    workspace = tmp_path / "workspace"
    monkeypatch.setenv("SFMAPI_BACKEND", "colmap_cli")
    monkeypatch.setenv("SFMAPI_COLMAP_EXECUTABLE", str(colmap_executable))
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
    backend_cls = _cpu_cli_backend_cls(cli_backend_cls)
    register_backend("colmap_cli", lambda: backend_cls(executable=colmap_executable))

    app = create_app()

    with TestClient(app) as client:
        capabilities = client.get("/v1/capabilities")
        assert capabilities.status_code == 200
        features = capabilities.json()["features"]
        assert features["features.extract.sift"] is True
        assert features["map.incremental"] is True

        project = client.post("/v1/projects", json={"name": "colmap-sample"}).json()
        project_id = project["project_id"]

        dataset_resp = client.post(
            f"/v1/projects/{project_id}/datasets",
            json={
                "name": "south-building-subset",
                "source": {
                    "kind": "local",
                    "root": str(colmap_sample_subset.image_root),
                    "recursive": False,
                },
                "camera_model": "SIMPLE_RADIAL",
                "intrinsics_mode": "single_camera",
                "respect_exif_orientation": True,
            },
        )
        assert dataset_resp.status_code == 201, dataset_resp.text
        dataset_id = dataset_resp.json()["dataset_id"]

        images_resp = client.post(
            f"/v1/datasets/{dataset_id}/images:batchCreate",
            json={
                "requests": [
                    {"name": name, "rel_path": name} for name in colmap_sample_subset.image_names
                ]
            },
        )
        assert images_resp.status_code == 201, images_resp.text
        assert len(images_resp.json()["images"]) == len(colmap_sample_subset.image_names)

        pipeline_resp = client.post(
            f"/v1/projects/{project_id}/pipelines/incremental",
            json={
                "dataset_id": dataset_id,
                "features": {"version": 1, "type": "sift"},
                "pairs": {"version": 1, "strategy": "exhaustive"},
                "matcher": {"version": 1, "type": "nn-mutual"},
                "verify": {"version": 1},
                "spec": {"version": 1, "kind": "incremental"},
            },
        )
        assert pipeline_resp.status_code == 202, pipeline_resp.text
        accepted = pipeline_resp.json()
        assert accepted["recon_id"]

        job_resp = client.get(f"/v1/jobs/{accepted['job_id']}")
        assert job_resp.status_code == 200
        job = job_resp.json()
        assert job["status"] == "succeeded", job

        snapshots_resp = client.get(f"/v1/reconstructions/{accepted['recon_id']}/snapshots")
        assert snapshots_resp.status_code == 200, snapshots_resp.text
        seqs = snapshots_resp.json()["seqs"]
        assert seqs

        summary_resp = client.get(
            f"/v1/reconstructions/{accepted['recon_id']}/snapshots/{seqs[-1]}/summary.json"
        )
        assert summary_resp.status_code == 200, summary_resp.text
        summary = summary_resp.json()
        assert summary["models"]
        assert max(item["num_reg_images"] for item in summary["models"]) >= 2


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.needs_colmap
@pytest.mark.needs_sample_data
@pytest.mark.slow
@pytest.mark.parametrize("cli_backend_cls", CLI_BACKENDS)
def test_sfmapi_http_colmap_backend_accepts_feature_artifact_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    colmap_executable: Path,
    colmap_sample_subset,
    cli_backend_cls,
):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from sceneapi.runtime import create_app, register_backend
    from sceneapi.testing import reset_runtime_for_tests_sync

    workspace = tmp_path / "workspace"
    monkeypatch.setenv("SFMAPI_BACKEND", "colmap_cli")
    monkeypatch.setenv("SFMAPI_COLMAP_EXECUTABLE", str(colmap_executable))
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
    backend_cls = _cpu_cli_backend_cls(cli_backend_cls)
    register_backend("colmap_cli", lambda: backend_cls(executable=colmap_executable))

    app = create_app()

    with TestClient(app) as client:
        project = client.post("/v1/projects", json={"name": "colmap-artifacts"}).json()
        project_id = project["project_id"]

        dataset_resp = client.post(
            f"/v1/projects/{project_id}/datasets",
            json={
                "name": "bicycle-subset",
                "source": {
                    "kind": "local",
                    "root": str(colmap_sample_subset.image_root),
                    "recursive": False,
                },
                "camera_model": "SIMPLE_RADIAL",
                "intrinsics_mode": "single_camera",
                "respect_exif_orientation": True,
            },
        )
        assert dataset_resp.status_code == 201, dataset_resp.text
        dataset_id = dataset_resp.json()["dataset_id"]

        images_resp = client.post(
            f"/v1/datasets/{dataset_id}/images:batchCreate",
            json={
                "requests": [
                    {"name": name, "rel_path": name} for name in colmap_sample_subset.image_names
                ]
            },
        )
        assert images_resp.status_code == 201, images_resp.text

        features_resp = client.post(
            f"/v1/datasets/{dataset_id}/features",
            json={"spec": {"version": 1, "type": "sift"}},
        )
        assert features_resp.status_code == 202, features_resp.text
        features_job_id = features_resp.json()["job_id"]
        features_job = client.get(f"/v1/jobs/{features_job_id}").json()
        assert features_job["status"] == "succeeded", features_job

        artifacts_resp = client.get(
            f"/v1/jobs/{features_job_id}/artifacts",
            params={"kind": "features.database"},
        )
        assert artifacts_resp.status_code == 200, artifacts_resp.text
        artifacts = artifacts_resp.json()["items"]
        assert len(artifacts) == 1
        feature_artifact = artifacts[0]
        assert feature_artifact["uri"]

        content_resp = client.get(f"/v1/artifacts/{feature_artifact['artifact_id']}/content")
        assert content_resp.status_code == 200, content_resp.text
        assert content_resp.content

        matches_resp = client.post(
            f"/v1/datasets/{dataset_id}/matches",
            json={
                "pairs": {"version": 1, "strategy": "exhaustive"},
                "matcher": {"version": 1, "type": "nn-mutual"},
                "input_artifacts": {
                    "features": {
                        "artifact_id": feature_artifact["artifact_id"],
                        "kind": "features.database",
                    }
                },
            },
        )
        assert matches_resp.status_code == 202, matches_resp.text
        matches_job_id = matches_resp.json()["job_id"]
        matches_job = client.get(f"/v1/jobs/{matches_job_id}").json()
        assert matches_job["status"] == "succeeded", matches_job

        match_artifacts_resp = client.get(
            f"/v1/jobs/{matches_job_id}/artifacts",
            params={"kind": "matches.database"},
        )
        assert match_artifacts_resp.status_code == 200, match_artifacts_resp.text
        assert match_artifacts_resp.json()["items"]


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.needs_pycolmap
@pytest.mark.needs_sample_data
@pytest.mark.slow
def test_sfmapi_http_pipeline_runs_pycolmap_backend_on_official_sample_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    colmap_sample_subset,
):
    pytest.importorskip("fastapi")
    pytest.importorskip("pycolmap")
    from fastapi.testclient import TestClient
    from sceneapi.runtime import create_app, register_backend
    from sceneapi.testing import reset_runtime_for_tests_sync

    workspace = tmp_path / "workspace"
    monkeypatch.setenv("SFMAPI_BACKEND", "colmap_pycolmap")
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
    register_backend("colmap_pycolmap", CpuPycolmapBackend)

    app = create_app()

    with TestClient(app) as client:
        capabilities = client.get("/v1/capabilities")
        assert capabilities.status_code == 200
        features = capabilities.json()["features"]
        assert features["features.extract.sift"] is True
        assert features["map.incremental"] is True

        project = client.post("/v1/projects", json={"name": "pycolmap-sample"}).json()
        project_id = project["project_id"]

        dataset_resp = client.post(
            f"/v1/projects/{project_id}/datasets",
            json={
                "name": "south-building-subset",
                "source": {
                    "kind": "local",
                    "root": str(colmap_sample_subset.image_root),
                    "recursive": False,
                },
                "camera_model": "SIMPLE_RADIAL",
                "intrinsics_mode": "single_camera",
                "respect_exif_orientation": True,
            },
        )
        assert dataset_resp.status_code == 201, dataset_resp.text
        dataset_id = dataset_resp.json()["dataset_id"]

        images_resp = client.post(
            f"/v1/datasets/{dataset_id}/images:batchCreate",
            json={
                "requests": [
                    {"name": name, "rel_path": name} for name in colmap_sample_subset.image_names
                ]
            },
        )
        assert images_resp.status_code == 201, images_resp.text

        pipeline_resp = client.post(
            f"/v1/projects/{project_id}/pipelines/incremental",
            json={
                "dataset_id": dataset_id,
                "features": {"version": 1, "type": "sift"},
                "pairs": {"version": 1, "strategy": "exhaustive"},
                "matcher": {"version": 1, "type": "nn-mutual"},
                "verify": {"version": 1},
                "spec": {
                    "version": 1,
                    "kind": "incremental",
                    "options": {"min_model_size": 2, "min_num_matches": 10},
                },
            },
        )
        assert pipeline_resp.status_code == 202, pipeline_resp.text
        accepted = pipeline_resp.json()
        assert accepted["recon_id"]

        job_resp = client.get(f"/v1/jobs/{accepted['job_id']}")
        assert job_resp.status_code == 200
        job = job_resp.json()
        assert job["status"] == "succeeded", job

        snapshots_resp = client.get(f"/v1/reconstructions/{accepted['recon_id']}/snapshots")
        assert snapshots_resp.status_code == 200, snapshots_resp.text
        seqs = snapshots_resp.json()["seqs"]
        assert seqs

        summary_resp = client.get(
            f"/v1/reconstructions/{accepted['recon_id']}/snapshots/{seqs[-1]}/summary.json"
        )
        assert summary_resp.status_code == 200, summary_resp.text
        summary = summary_resp.json()
        assert summary["models"]
        assert max(item["num_reg_images"] for item in summary["models"]) >= 2
