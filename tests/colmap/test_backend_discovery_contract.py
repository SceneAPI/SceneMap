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


def _fake_colmap(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def _schema(command: str) -> dict[str, Any]:
    return {
        "command": command,
        "available": True,
        "schema_source": "test",
        "option_count": 2,
        "options": [
            {
                "name": "database_path",
                "flags": ["--database_path"],
                "takes_value": True,
                "schema": {"type": "string", "format": "path"},
            },
            {
                "name": "SiftExtraction.peak_threshold",
                "flags": ["--SiftExtraction.peak_threshold"],
                "takes_value": True,
                "schema": {"type": "number"},
            },
        ],
    }


@pytest.mark.parametrize("backend_cls", PROVIDER_BACKENDS)
def test_colmap_commands_are_backend_actions_not_capabilities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, backend_cls
) -> None:
    backend = backend_cls(executable=_fake_colmap(tmp_path / "colmap"))
    monkeypatch.setattr(backend, "colmap_command_schema", _schema)

    actions = backend.list_backend_actions(include_schemas=True)
    action_ids = {action["action_id"] for action in actions}

    assert "colmap.feature_extractor" in action_ids
    assert not any(capability.startswith("colmap.") for capability in backend.capabilities())
    feature_action = next(
        action for action in actions if action["action_id"] == "colmap.feature_extractor"
    )
    assert "database_path" in feature_action["input_schema"]["properties"]


@pytest.mark.parametrize("backend_cls", PROVIDER_BACKENDS)
def test_backend_config_schemas_cover_backend_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, backend_cls
) -> None:
    backend = backend_cls(executable=_fake_colmap(tmp_path / "colmap"))
    monkeypatch.setattr(backend, "colmap_command_schema", _schema)

    rows = backend.list_backend_config_schemas(include_schemas=True)
    feature = next(row for row in rows if row["config_id"] == "colmap.features.sift")

    assert feature["capability"] == "features.extract.sift"
    assert feature["stage"] == "features"
    assert "database_path" not in feature["option_schema"]["properties"]
    assert "SiftExtraction.peak_threshold" in feature["option_schema"]["properties"]


@pytest.mark.parametrize("backend_cls", PROVIDER_BACKENDS)
def test_backend_discovery_contract_is_enforced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, backend_cls
) -> None:
    contract = pytest.importorskip("sceneapi.backends")
    backend = backend_cls(executable=_fake_colmap(tmp_path / "colmap"))
    monkeypatch.setattr(backend, "colmap_command_schema", _schema)

    contract.assert_backend_contract(backend)
