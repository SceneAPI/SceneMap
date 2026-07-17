from __future__ import annotations

import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from sceneapi_map.colmap.pycolmap.plugin import (
    PLUGIN_MANIFEST,
    get_plugin_manifest,
    plugin,
    register,
)
from sceneapi_map.colmap.pycolmap_backend import PycolmapBackend

# Union of sfmapi_pycolmap's test_plugin.py + test_plugin_contract.py
# (they overlapped heavily; near-duplicate assertions merged).

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_plugin_manifest_ids_match_hub_registry_expectations() -> None:
    manifest = plugin.get_plugin_manifest()

    assert manifest is PLUGIN_MANIFEST
    assert manifest is get_plugin_manifest()
    assert manifest["plugin_id"] == "pycolmap"
    # Manifest identity was re-pointed at this merged repo (see README
    # migration notes): package/repo coordinates name sceneapi-map
    # while the plugin id stays pycolmap.
    assert manifest["entry_points"] == ["sceneapi_map.colmap.pycolmap.plugin:plugin"]
    assert manifest["package_name"] == "sceneapi-map"
    assert [provider["provider_id"] for provider in manifest["providers"]] == ["colmap_pycolmap"]


def test_plugin_manifest_validates_against_sfm_hub_if_available() -> None:
    models = pytest.importorskip("sfm_hub.models")

    manifest = models.PluginManifest.model_validate(PLUGIN_MANIFEST)

    assert manifest.plugin_id == "pycolmap"
    assert manifest.provider_ids() == ["colmap_pycolmap"]


def test_declared_entry_point_loads_plugin_object() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entry_point_value = pyproject["project"]["entry-points"]["sceneapi.backends"]["pycolmap"]

    assert entry_point_value == "sceneapi_map.colmap.pycolmap.plugin:plugin"
    loaded = metadata.EntryPoint(
        name="pycolmap",
        value=entry_point_value,
        group="sceneapi.backends",
    ).load()

    assert loaded is plugin
    assert loaded.get_plugin_manifest()["plugin_id"] == "pycolmap"


def test_plugin_registers_pycolmap_backend_factory() -> None:
    registered: dict[str, Any] = {}

    plugin.register(registered.__setitem__)

    assert registered == {"colmap_pycolmap": PycolmapBackend}
    assert isinstance(registered["colmap_pycolmap"](), PycolmapBackend)


def test_module_level_register_hook_registers_pycolmap_backend_factory() -> None:
    registered: dict[str, Any] = {}

    register(lambda name, factory: registered.setdefault(name, factory))

    assert registered == {"colmap_pycolmap": PycolmapBackend}
