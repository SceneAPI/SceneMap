from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scenemap.colmap.cli.backend import ColmapCliBackend as CliColmapCliBackend
from scenemap.colmap.pycolmap.backend import ColmapCliBackend as PycolmapColmapCliBackend

# sfmapi_pycolmap and sfmapi_colmap_cli carried this file as import-only
# variants over their own ColmapCliBackend implementations; parametrize
# to keep both covered.
PROVIDER_BACKENDS = [
    pytest.param(PycolmapColmapCliBackend, id="pycolmap"),
    pytest.param(CliColmapCliBackend, id="cli"),
]


def _discovery_backend_cls(base: type) -> type:
    class DiscoveryBackend(base):
        def colmap_command_schema(self, command: str) -> dict[str, Any]:
            return {
                "command": command,
                "available": True,
                "schema_source": "test",
                "option_count": 1,
                "options": [
                    {
                        "name": "SiftExtraction.peak_threshold",
                        "flags": ["--SiftExtraction.peak_threshold"],
                        "takes_value": True,
                        "schema": {"type": "number"},
                    },
                ],
            }

    return DiscoveryBackend


def _fake_colmap(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


@pytest.mark.parametrize("backend_cls", PROVIDER_BACKENDS)
def test_sfmapi_http_discovery_surfaces_colmap_actions_and_config_schemas(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend_cls,
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from sceneapi.runtime import register_backend
    from sceneapi.testing import reset_runtime_for_tests_sync

    fake_colmap = _fake_colmap(tmp_path / "colmap")
    monkeypatch.setenv("SFMAPI_BACKEND", "colmap_cli")
    monkeypatch.setenv("SFMAPI_MCP_MODE", "off")
    from sceneapi.runtime import create_app

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
        workspace_root=tmp_path / "workspace",
    )
    discovery_cls = _discovery_backend_cls(backend_cls)
    register_backend("colmap_cli", lambda: discovery_cls(executable=fake_colmap))

    with TestClient(create_app()) as client:
        capabilities = client.get("/v1/capabilities").json()
        assert capabilities["features"]["features.extract.sift"] is True
        assert capabilities["features"]["backend.actions"] is True
        assert capabilities["features"]["backend.config_schemas"] is True

        backend = client.get("/v1/backend").json()
        assert backend["name"] == "colmap_cli"
        assert backend["action_count"] > 0
        assert backend["config_schema_count"] > 0

        actions = client.get("/v1/backend/actions?include_schemas=true").json()["items"]
        feature_action = next(
            action for action in actions if action["action_id"] == "colmap.feature_extractor"
        )
        assert "SiftExtraction.peak_threshold" in feature_action["input_schema"]["properties"]

        config_schemas = client.get("/v1/backend/config-schemas").json()["items"]
        feature_config = next(
            row for row in config_schemas if row["config_id"] == "colmap.features.sift"
        )
        assert "SiftExtraction.peak_threshold" in feature_config["option_schema"]["properties"]
