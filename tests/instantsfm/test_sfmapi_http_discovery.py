from __future__ import annotations

from pathlib import Path

import pytest

from scenemap.instantsfm.backend import InstantSfMBackend


def _fake_instantsfm(root: Path) -> Path:
    (root / "instantsfm").mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname='instantsfm'\n", encoding="utf-8")
    return root


def test_sfmapi_http_discovery_surfaces_instantsfm_actions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from sceneapi.runtime import register_backend
    from sceneapi.testing import reset_runtime_for_tests_sync

    root = _fake_instantsfm(tmp_path / "InstantSfM")
    monkeypatch.setenv("SFMAPI_BACKEND", "instantsfm")
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
    register_backend("instantsfm", lambda: InstantSfMBackend(root))

    with TestClient(create_app()) as client:
        capabilities = client.get("/v1/capabilities").json()
        assert capabilities["backend"]["name"] == "instantsfm"
        assert capabilities["features"]["backend.actions"] is True
        # config_schemas now true: instantsfm.mapping.global is published
        # for the map.global stage's backend_options envelope.
        assert capabilities["features"]["backend.config_schemas"] is True

        backend = client.get("/v1/backend").json()
        assert backend["name"] == "instantsfm"
        assert backend["action_count"] > 0
        assert backend["config_schema_count"] == 1

        actions = client.get("/v1/backend/actions?include_schemas=true&page_size=50").json()[
            "items"
        ]
        action_ids = {action["action_id"] for action in actions}
        assert "instantsfm.runPipeline" in action_ids
        assert "instantsfm.extractFeatures" in action_ids
        pipeline = next(
            action for action in actions if action["action_id"] == "instantsfm.runPipeline"
        )
        assert "data_path" in pipeline["input_schema"]["properties"]

        schemas = client.get("/v1/backend/config-schemas").json()["items"]
        assert [s["config_id"] for s in schemas] == ["instantsfm.mapping.global"]
