"""Plugin / manifest / entry-point + provisioning tests for MapAnything."""

from __future__ import annotations

import tomllib
from importlib import import_module
from pathlib import Path

from sceneapi_map.mapanything.backend import (
    APACHE_WEIGHTS,
    CC_BY_NC_WEIGHTS,
    WEIGHTS_ENV_VAR,
    MapAnythingBackend,
)
from sceneapi_map.mapanything.plugin import get_plugin_manifest, plugin
from sceneapi_map.mapanything.provisioning import provision


def test_plugin_manifest_matches_hub_contract() -> None:
    manifest = get_plugin_manifest()

    assert manifest["plugin_id"] == "mapanything"
    assert manifest["entry_points"] == ["sceneapi_map.mapanything.plugin:plugin"]
    assert [provider["provider_id"] for provider in manifest["providers"]] == ["mapanything"]
    assert manifest["package_name"] == "sceneapi-map"
    assert manifest["github_url"] == "https://github.com/SceneAPI/SceneMap.git"
    assert manifest["runtime_modes"]["uv"]["package"] == "sceneapi-map"
    assert manifest["runtime_modes"]["docker"] is None
    assert manifest["compatibility"]["torch"]["device"] == "cuda"


def test_manifest_declares_feed_forward_capability() -> None:
    manifest = get_plugin_manifest()

    assert manifest["capabilities"] == ["map.feed_forward"]
    assert manifest["providers"][0]["capabilities"] == ["map.feed_forward"]
    assert manifest["backend_actions"] == []
    assert manifest["config_schemas"] == []
    assert manifest["artifact_contracts"] == []
    assert manifest["trust_tier"] == "community"


def test_manifest_documents_weights_license_opt_in() -> None:
    manifest = get_plugin_manifest()

    # Apache is the default; the non-commercial variant + its env flag are named.
    assert APACHE_WEIGHTS in manifest["description"]
    assert CC_BY_NC_WEIGHTS in manifest["description"]
    assert WEIGHTS_ENV_VAR in manifest["description"]
    assert manifest["licenses"] == [{"name": "Apache-2.0"}]
    assert manifest["compatibility"]["source_build"]["default_weights"] == APACHE_WEIGHTS
    assert manifest["compatibility"]["source_build"]["weights_env"] == WEIGHTS_ENV_VAR


def test_pyproject_declares_backend_entry_point_and_extra() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    entry_points = pyproject["project"]["entry-points"]["sceneapi.backends"]
    assert entry_points["mapanything"] == "sceneapi_map.mapanything.plugin:plugin"
    assert pyproject["project"]["version"] == "0.3.0"
    # Heavy engine deps live in the opt-in extra (deferred to provisioning).
    assert "mapanything" in pyproject["project"]["optional-dependencies"]
    extra = pyproject["project"]["optional-dependencies"]["mapanything"]
    assert any(dep.startswith("torch") for dep in extra)


def test_configured_entry_point_imports_plugin_object() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    target = pyproject["project"]["entry-points"]["sceneapi.backends"]["mapanything"]
    module_name, object_name = target.split(":", maxsplit=1)

    loaded = getattr(import_module(module_name), object_name)

    assert loaded is plugin
    assert loaded.get_plugin_manifest()["plugin_id"] == "mapanything"


def test_plugin_registers_backend_factory() -> None:
    registered: dict[str, object] = {}

    plugin.register(lambda name, factory, **_: registered.update({name: factory}))

    factory = registered["mapanything"]
    assert callable(factory)
    assert isinstance(factory(), MapAnythingBackend)


def test_provisioning_dry_run_plans_engine_and_weights_steps() -> None:
    plan = provision(dry_run=True)

    assert plan["available"] is True
    assert plan["provisioned"] is False
    step_names = {step["name"] for step in plan["steps"]}
    assert {"torch_runtime", "mapanything_engine", "mapanything_weights"} <= step_names
    # Dry run never touches the network / installs anything.
    assert all(step["status"] == "planned" for step in plan["steps"])
