from __future__ import annotations

import tomllib
from importlib import import_module
from pathlib import Path

from sceneapi_map.instantsfm.backend import InstantSfMBackend
from sceneapi_map.instantsfm.plugin import get_plugin_manifest, plugin


def test_plugin_manifest_matches_hub_contract() -> None:
    manifest = get_plugin_manifest()

    assert manifest["plugin_id"] == "instantsfm"
    assert manifest["entry_points"] == ["sceneapi_map.instantsfm.plugin:plugin"]
    assert [provider["provider_id"] for provider in manifest["providers"]] == ["instantsfm"]
    assert manifest["compatibility"]["torch"]["device"] == "cuda"
    assert manifest["runtime_modes"]["container_service"]["execution"]["gpu"] == "required"
    assert (
        manifest["runtime_modes"]["container_service"]["image"]["build"]["args"]["TORCH_DEVICE"]
        == "cuda"
    )
    assert (
        "TORCH_CUDA_ARCH_LIST"
        in manifest["runtime_modes"]["container_service"]["image"]["build"]["args"]
    )


def test_pyproject_declares_sfmapi_backend_entry_point() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    entry_points = pyproject["project"]["entry-points"]["sceneapi.backends"]
    assert entry_points["instantsfm"] == "sceneapi_map.instantsfm.plugin:plugin"
    scripts = pyproject["project"]["scripts"]
    assert scripts["sfmapi-instantsfm-service"] == "sceneapi_map.instantsfm.container_service:main"


def test_configured_entry_point_imports_plugin_object() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    target = pyproject["project"]["entry-points"]["sceneapi.backends"]["instantsfm"]
    module_name, object_name = target.split(":", maxsplit=1)

    loaded = getattr(import_module(module_name), object_name)

    assert loaded is plugin
    assert loaded.get_plugin_manifest()["plugin_id"] == "instantsfm"


def test_manifest_container_build_target_exists() -> None:
    manifest = get_plugin_manifest()
    dockerfile = manifest["runtime_modes"]["container_service"]["image"]["build"]["dockerfile"]

    assert Path(str(dockerfile)).is_file()
    dockerfile_text = Path(str(dockerfile)).read_text(encoding="utf-8")
    assert "libcudss0-dev-cuda-12" in dockerfile_text


def test_plugin_registers_backend_factory() -> None:
    registered: dict[str, object] = {}

    plugin.register(lambda name, factory: registered.update({name: factory}))

    factory = registered["instantsfm"]
    assert callable(factory)
    assert isinstance(factory(), InstantSfMBackend)


def test_manifest_declares_map_global_capability(tmp_path: Path) -> None:
    # InstantSfM is a *global* SfM engine. It backs exactly one portable
    # capability -- `map.global` -- via the run_mapping path-staging
    # adapter. Feature extraction stays action-only (fused extract+match
    # in scripts.feat). It also publishes one config-schema
    # `instantsfm.mapping.global` for the portable `map.global` stage's
    # `backend_options` envelope. Everything else is exposed via
    # instantsfm.* backend actions; artifact contracts stay empty.
    manifest = get_plugin_manifest()
    assert manifest["capabilities"] == ["map.global"]
    assert manifest["providers"][0]["capabilities"] == ["map.global"]
    assert manifest["config_schemas"] == ["instantsfm.mapping.global"]
    assert manifest["artifact_contracts"] == []
    assert manifest["backend_actions"] == ["instantsfm.*"]

    # The backend agrees when the InstantSfM checkout is resolvable.
    root = tmp_path / "InstantSfM"
    (root / "instantsfm").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='instantsfm'\n", encoding="utf-8")
    assert InstantSfMBackend(root).capabilities() == {"map.global"}

    # ...and advertises nothing when the checkout cannot be found.
    assert InstantSfMBackend(tmp_path / "missing").capabilities() == set()
