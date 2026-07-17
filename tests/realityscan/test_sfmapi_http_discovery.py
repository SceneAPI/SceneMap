from __future__ import annotations

from pathlib import Path

import pytest

from sceneapi_map.realityscan.backend import RealityScanCliBackend


def _fake_realityscan(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def test_sfmapi_http_discovery_surfaces_realityscan_actions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from sceneapi.runtime import register_backend
    from sceneapi.testing import reset_runtime_for_tests_sync

    fake_exe = _fake_realityscan(tmp_path / "RealityCapture_3.0" / "RealityCapture.exe")
    monkeypatch.setenv("SFMAPI_BACKEND", "realityscan_cli")
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
    register_backend("realityscan_cli", lambda: RealityScanCliBackend(fake_exe))

    with TestClient(create_app()) as client:
        capabilities = client.get("/v1/capabilities").json()
        assert capabilities["backend"]["name"] == "realityscan_cli"
        assert capabilities["features"]["backend.actions"] is True
        assert capabilities["features"]["backend.config_schemas"] is False

        backend = client.get("/v1/backend").json()
        assert backend["name"] == "realityscan_cli"
        assert backend["action_count"] > 0
        assert backend["config_schema_count"] == 0

        actions = client.get("/v1/backend/actions?include_schemas=true&page_size=500").json()[
            "items"
        ]
        action_ids = {action["action_id"] for action in actions}
        assert "realityscan.reconstructImageFolder" in action_ids
        assert "realityscan.align" in action_ids
        # Headline mesh / texture / ortho verbs and canned workflows surface
        # through HTTP discovery now that the catalog is no longer SfM-only.
        assert "realityscan.calculateOrthoProjection" in action_ids
        assert "realityscan.calculateTexture" in action_ids
        assert "realityscan.reconstructToTexturedMesh" in action_ids
        # Genuine UI-only / upload verbs stay out of the catalog.
        assert "realityscan.uploadToSketchfab" not in action_ids
        assert "realityscan.dtmClassify" not in action_ids
        reconstruct = next(
            action
            for action in actions
            if action["action_id"] == "realityscan.reconstructImageFolder"
        )
        assert "image_folder" in reconstruct["input_schema"]["properties"]

        assert client.get("/v1/backend/config-schemas").json()["items"] == []

        contracts = client.get("/v1/backend/artifact-contracts").json()["items"]
        contract_ids = {contract["contract_id"] for contract in contracts}
        assert contract_ids == {
            "realityscan.registration",
            "realityscan.sparse_point_cloud",
        }
        for contract in contracts:
            assert contract["capability"] is None
