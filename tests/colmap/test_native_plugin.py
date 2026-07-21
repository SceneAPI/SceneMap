from __future__ import annotations

import json
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from scenemap.colmap.native.backend import ColmapCliBackend
from scenemap.colmap.native.cpp_inmemory_backend import CppInmemoryBackend
from scenemap.colmap.native.cpp_native_backend import CppNativeBackend
from scenemap.colmap.native.plugin import get_plugin_manifest, manifest, plugin, register
from scenemap.colmap.pycolmap_backend import PycolmapBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PROVIDER_FACTORIES = {
    "colmap_cli": ColmapCliBackend,
    "colmap_pycolmap": PycolmapBackend,
    "colmap_cpp_native": CppNativeBackend,
    "colmap_cpp_inmemory": CppInmemoryBackend,
}


def test_plugin_manifest_validates_against_sfm_hub_contract() -> None:
    models = pytest.importorskip("sfm_hub.models")

    validated = models.PluginManifest.model_validate(get_plugin_manifest())

    assert validated.plugin_id == "colmap_native"
    # Manifest identity was re-pointed at this merged repo (see README
    # migration notes): package/repo coordinates name scenemap
    # while the plugin id stays colmap_native.
    assert validated.package_name == "scenemap"
    assert validated.entry_points == ["scenemap.colmap.native.plugin:plugin"]
    assert set(validated.provider_ids()) == set(EXPECTED_PROVIDER_FACTORIES)


def test_plugin_manifest_matches_bundled_registry_manifest_when_available() -> None:
    registry_manifest = (
        REPO_ROOT.parent
        / "sfmapi"
        / "sfm_hub"
        / "registry"
        / "backends"
        / "colmap_native"
        / "manifest.json"
    )
    if not registry_manifest.is_file():
        pytest.skip("sfmapi bundled registry manifest is not available beside this checkout")

    expected = json.loads(registry_manifest.read_text(encoding="utf-8"))
    actual = get_plugin_manifest()

    assert actual["plugin_id"] == expected["plugin_id"] == "colmap_native"
    assert actual["entry_points"] == expected["entry_points"]
    assert actual["providers"] == expected["providers"]
    assert actual["capabilities"] == expected["capabilities"]
    assert actual["backend_actions"] == expected["backend_actions"]
    assert actual["config_schemas"] == expected["config_schemas"]


def test_declared_entry_point_loads_plugin_object() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entry_point_value = pyproject["project"]["entry-points"]["sceneapi.backends"]["colmap_native"]

    assert entry_point_value == "scenemap.colmap.native.plugin:plugin"
    loaded = metadata.EntryPoint(
        name="colmap_native",
        value=entry_point_value,
        group="sceneapi.backends",
    ).load()

    assert loaded is plugin
    assert loaded.manifest is manifest
    assert loaded.get_plugin_manifest()["plugin_id"] == "colmap_native"


def test_plugin_registers_all_provider_factories() -> None:
    registered: dict[str, Any] = {}

    register(registered.__setitem__)

    assert registered == EXPECTED_PROVIDER_FACTORIES


def test_plugin_object_register_method_registers_all_provider_factories() -> None:
    registered: dict[str, Any] = {}

    plugin.register(registered.__setitem__)

    assert registered == EXPECTED_PROVIDER_FACTORIES
