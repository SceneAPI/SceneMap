from __future__ import annotations

import tomllib
from importlib import import_module
from pathlib import Path

from sceneapi_map.spheresfm.backend import SPHERESFM_CAPABILITIES, SphereSfMBackend
from sceneapi_map.spheresfm.plugin import get_plugin_manifest, plugin


def test_plugin_manifest_matches_hub_contract() -> None:
    manifest = get_plugin_manifest()

    assert manifest["plugin_id"] == "spheresfm"
    assert manifest["entry_points"] == ["sceneapi_map.spheresfm.plugin:plugin"]
    assert [provider["provider_id"] for provider in manifest["providers"]] == ["spheresfm"]


def test_pyproject_declares_sfmapi_backend_entry_point() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    entry_points = pyproject["project"]["entry-points"]["sceneapi.backends"]
    assert entry_points["spheresfm"] == "sceneapi_map.spheresfm.plugin:plugin"


def test_configured_entry_point_imports_plugin_object() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    target = pyproject["project"]["entry-points"]["sceneapi.backends"]["spheresfm"]
    module_name, object_name = target.split(":", maxsplit=1)

    loaded = getattr(import_module(module_name), object_name)

    assert loaded is plugin
    assert loaded.get_plugin_manifest()["plugin_id"] == "spheresfm"


def test_plugin_registers_backend_factory() -> None:
    registered: dict[str, object] = {}

    plugin.register(lambda name, factory: registered.update({name: factory}))

    factory = registered["spheresfm"]
    assert callable(factory)
    assert isinstance(factory(), SphereSfMBackend)


def test_manifest_capabilities_match_backend(tmp_path: Path) -> None:
    # The manifest must not advertise portable capabilities the backend
    # cannot actually back with a wrapper method. SphereSfM is a COLMAP
    # fork, so the manifest mirrors the full SPHERESFM_CAPABILITIES set
    # (features, every matcher, all three mapping kinds, BA incl. rig,
    # triangulation, relocalization, merge, export, georegistration).
    manifest = get_plugin_manifest()
    expected = set(SPHERESFM_CAPABILITIES)
    assert set(manifest["capabilities"]) == expected
    assert set(manifest["providers"][0]["capabilities"]) == expected

    fake_exe = tmp_path / "colmap.exe"
    fake_exe.write_text("", encoding="utf-8")
    backend = SphereSfMBackend(fake_exe)
    assert backend.capabilities() == expected

    # Declared artifact-contract ids must resolve to real contracts.
    contract_ids = {row["contract_id"] for row in backend.list_backend_artifact_contracts()}
    assert set(manifest["artifact_contracts"]) == contract_ids
