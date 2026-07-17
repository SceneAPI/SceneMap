from __future__ import annotations

from pathlib import Path

import pytest

from sceneapi_map.spheresfm.backend import SphereSfMBackend


def _fake_colmap(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def test_sfmapi_http_discovery_surfaces_spheresfm_actions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from sceneapi.runtime import register_backend
    from sceneapi.testing import reset_runtime_for_tests_sync

    exe = _fake_colmap(tmp_path / "colmap.exe")
    monkeypatch.setenv("SFMAPI_BACKEND", "spheresfm")
    monkeypatch.setenv("SFMAPI_MCP_MODE", "off")
    from sceneapi.runtime import create_app

    reset_runtime_for_tests_sync(
        ephemeral=True,
        # Merge adaptation: the unified package installs all six entry
        # points, so lifespan autoload would re-register the real providers
        # over this test's fake. Pin autoload off (the core's documented
        # test convention; the colmap suite already does this).
        auto_load_backend_plugins=False,
        db_url="sqlite+aiosqlite:///file::memory:?cache=shared&uri=true",
        blob_backend="memory",
        queue_backend="inline",
        inline_tasks=True,
        workspace_root=tmp_path / "workspace",
    )
    register_backend("spheresfm", lambda: SphereSfMBackend(exe))

    with TestClient(create_app()) as client:
        capabilities = client.get("/v1/capabilities").json()
        assert capabilities["backend"]["name"] == "spheresfm"
        assert capabilities["features"]["backend.actions"] is True
        # SphereSfM now ships portable stage capabilities + their option
        # schemas + artifact contracts.
        assert capabilities["features"]["backend.config_schemas"] is True
        assert capabilities["features"]["backend.artifact_contracts"] is True
        assert capabilities["features"]["features.extract.sift"] is True
        assert capabilities["features"]["map.spherical"] is True
        assert capabilities["features"]["projection.cubemap_rig"] is True

        backend = client.get("/v1/backend").json()
        assert backend["name"] == "spheresfm"
        assert backend["action_count"] > 0
        assert backend["config_schema_count"] == 3

        actions = client.get("/v1/backend/actions?include_schemas=true&page_size=100").json()[
            "items"
        ]
        action_ids = {action["action_id"] for action in actions}
        assert "spheresfm.reconstructPanoramaFolder" in action_ids
        assert "spheresfm.convertToCubemap" in action_ids
        assert "spheresfm.colmap.sphere_cubic_reprojecer" in action_ids
        reconstruct = next(
            action
            for action in actions
            if action["action_id"] == "spheresfm.reconstructPanoramaFolder"
        )
        assert "matching_mode" in reconstruct["input_schema"]["properties"]

        config_ids = {
            row["config_id"] for row in client.get("/v1/backend/config-schemas").json()["items"]
        }
        assert config_ids == {
            "spheresfm.features.sift",
            "spheresfm.pairs.exhaustive",
            "spheresfm.mapping.spherical",
        }
        contract_ids = {
            row["contract_id"]
            for row in client.get("/v1/backend/artifact-contracts").json()["items"]
        }
        assert contract_ids == {
            "spheresfm.matches.database",
            "spheresfm.reconstruction.spherical",
        }
