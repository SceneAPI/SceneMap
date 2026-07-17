from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from sceneapi_map.colmap.cli.backend import ColmapCliBackend
from sceneapi_map.colmap.cli.plugin import MANIFEST, plugin


def test_plugin_manifest_ids_match_hub_registry_expectations() -> None:
    assert MANIFEST["plugin_id"] == "colmap_cli"
    # Manifest identity was re-pointed at this merged repo (see README
    # migration notes): package/repo coordinates name sceneapi-map
    # while the plugin id stays colmap_cli.
    assert MANIFEST["entry_points"] == ["sceneapi_map.colmap.cli.plugin:plugin"]
    assert MANIFEST["package_name"] == "sceneapi-map"
    assert [provider["provider_id"] for provider in MANIFEST["providers"]] == ["colmap_cli"]
    assert plugin.backend_name == "colmap_cli"


def test_plugin_manifest_validates_against_sfmapi_model() -> None:
    models = pytest.importorskip("sfm_hub.models")

    manifest = models.PluginManifest.model_validate(MANIFEST)

    assert manifest.plugin_id == "colmap_cli"
    assert manifest.provider_ids() == ["colmap_cli"]


def test_plugin_entry_point_target_imports_and_discovers_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery = pytest.importorskip("sfm_hub.discovery")

    entry_point = metadata.EntryPoint(
        name="colmap_cli",
        value="sceneapi_map.colmap.cli.plugin:plugin",
        group="sceneapi.backends",
    )

    class FakeEntryPoints(list[metadata.EntryPoint]):
        def select(self, *, group: str) -> list[metadata.EntryPoint]:
            # Merge adaptation: the renamed core reads the primary
            # ``sceneapi.backends`` group and, for one release, the legacy
            # ``sfmapi.backends`` group too. Serve the plugin from the
            # primary group only.
            assert group in ("sceneapi.backends", "sfmapi.backends")
            return list(self) if group == "sceneapi.backends" else []

    monkeypatch.setattr(discovery.metadata, "entry_points", lambda: FakeEntryPoints([entry_point]))

    discovered = discovery.discover_plugins(load=True)

    assert discovered[0].plugin_id == "colmap_cli"
    assert discovered[0].entry_point == "sceneapi_map.colmap.cli.plugin:plugin"
    assert discovered[0].manifest is not None
    assert discovered[0].manifest.provider_ids() == ["colmap_cli"]


def test_plugin_registers_colmap_backend_factory() -> None:
    registered: dict[str, Any] = {}

    plugin.register(lambda name, factory: registered.setdefault(name, factory))

    assert list(registered) == ["colmap_cli"]
    assert isinstance(registered["colmap_cli"](), ColmapCliBackend)


def test_pyproject_declares_sfmapi_backend_entry_point() -> None:
    tomllib = pytest.importorskip("tomllib")
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    entry_points = pyproject["project"]["entry-points"]["sceneapi.backends"]
    # The unified pyproject now declares all three COLMAP entry points;
    # this provider's row must keep its historical name.
    assert entry_points["colmap_cli"] == "sceneapi_map.colmap.cli.plugin:plugin"
