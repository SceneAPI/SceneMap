"""sceneapi plugin hub entry point for the MapAnything feed-forward backend."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

# `MapAnythingPlugin` is this module's exported per-plugin alias of the
# canonical sceneapi.backends.Plugin, so downstream consumers keep working.
from sceneapi.backends import Plugin as MapAnythingPlugin

from .backend import (
    APACHE_WEIGHTS,
    CC_BY_NC_WEIGHTS,
    WEIGHTS_ENV_VAR,
    MapAnythingBackend,
    backend_factory,
)


class ProviderManifestDict(TypedDict):
    provider_id: str
    display_name: str
    description: str
    capabilities: list[str]
    backend_actions: list[str]
    priority_hint: int


class PluginManifestDict(TypedDict):
    schema_version: int
    plugin_id: str
    display_name: str
    description: str
    package_name: str
    github_url: str
    entry_points: list[str]
    providers: list[ProviderManifestDict]
    runtime_modes: dict[str, Any]
    capabilities: list[str]
    backend_actions: list[str]
    config_schemas: list[str]
    artifact_contracts: list[str]
    licenses: list[dict[str, str]]
    upstream_projects: list[dict[str, str]]
    compatibility: dict[str, Any]
    conformance: dict[str, str]
    trust_tier: str


# MapAnything needs a CUDA Torch runtime; mirror the torch-backed family
# runtime block (validated by sfm_hub's TorchRuntime).
TORCH_RUNTIME = {
    "policy": "required",
    "device": "cuda",
    "index_url": "https://download.pytorch.org/whl/cu128",
    "cpu_index_url": "https://download.pytorch.org/whl/cpu",
    "packages": ["torch", "torchvision"],
    "install_env": {
        "TORCH_DEVICE": "cuda",
        "TORCH_INDEX_URL": "https://download.pytorch.org/whl/cu128",
        "TORCH_CPU_INDEX_URL": "https://download.pytorch.org/whl/cpu",
        "TORCH_PACKAGES": "torch torchvision",
    },
}


manifest: PluginManifestDict = {
    "schema_version": 1,
    "plugin_id": "mapanything",
    "display_name": "MapAnything",
    "description": (
        "MapAnything (Meta Reality Labs + CMU, arXiv 2509.13414): a single "
        "feed-forward transformer that regresses factored metric 3D geometry "
        "from raw views with NO correspondences. Backs the portable "
        "`map.feed_forward` capability via the sceneapi-io Mapper contract. "
        "Wrapper + SDK material is Apache-2.0. Weights are opt-in by license: "
        f"the DEFAULT is the Apache-2.0 variant `{APACHE_WEIGHTS}`; the better "
        f"CC-BY-NC-4.0 variant `{CC_BY_NC_WEIGHTS}` is NON-COMMERCIAL and is "
        "selected only by explicit opt-in (a MappingOptions.extra `weights` "
        f"key or the `{WEIGHTS_ENV_VAR}` env var). That non-commercial term is "
        "upstream's and binds whoever runs those weights, not added by sceneapi."
    ),
    "package_name": "sceneapi-map",
    "github_url": "https://github.com/SceneAPI/SceneMap.git",
    "entry_points": ["sceneapi_map.mapanything.plugin:plugin"],
    "providers": [
        {
            "provider_id": "mapanything",
            "display_name": "MapAnything",
            "description": (
                "Feed-forward metric 3D reconstruction. Registers all views "
                "in one pass; emits dense per-view pointmaps + confidence and a "
                "fused sparse cloud. Optional calibration / pose / depth priors."
            ),
            # One portable capability, backed by the sceneapi-io Mapper contract
            # (MapAnythingBackend implements traits()/map()). No backend actions.
            "capabilities": ["map.feed_forward"],
            "backend_actions": [],
            "priority_hint": 50,
        }
    ],
    "runtime_modes": {
        # Heavy engine (git-only model package + CUDA Torch + weights) is
        # deferred to the install-time provisioning hook; the uv mode installs
        # the wrapper package and then runs `sceneapi_map.mapanything.provisioning`.
        "uv": {
            "source": "git",
            "url": "https://github.com/SceneAPI/SceneMap.git",
            "ref": "main",
            "package": "sceneapi-map",
        },
        "docker": None,
    },
    "capabilities": ["map.feed_forward"],
    "backend_actions": [],
    "config_schemas": [],
    "artifact_contracts": [],
    "licenses": [{"name": "Apache-2.0"}],
    "upstream_projects": [
        {
            "name": "MapAnything",
            "url": "https://github.com/facebookresearch/map-anything",
            # Apache-2.0 is the DEFAULT weights variant this provider selects;
            # the CC-BY-NC-4.0 weights are opt-in (see description).
            "license": "Apache-2.0",
        }
    ],
    "compatibility": {
        "sfmapi": ">=0.0.1",
        "python": ">=3.12,<3.13",
        "os": ["windows", "linux"],
        "cuda": "required",
        "torch": TORCH_RUNTIME,
        "source_build": {
            # The `mapanything` model package is not on PyPI; it is installed
            # from git (`pip install -e ".[all]"`) by the provisioning hook,
            # which also fetches the Hugging Face weights on first run.
            "upstream_repo": "https://github.com/facebookresearch/map-anything",
            "install": 'pip install -e ".[all]"',
            "default_weights": APACHE_WEIGHTS,
            "weights_env": WEIGHTS_ENV_VAR,
        },
    },
    "conformance": {"status": "not_run", "suite": "sfmapi-bench"},
    "trust_tier": "community",
}


def get_plugin_manifest() -> PluginManifestDict:
    return manifest


def register(register_backend: Callable[..., None]) -> None:
    provider_ids = [str(provider["provider_id"]) for provider in manifest["providers"]]
    try:
        register_backend("mapanything", backend_factory, providers=provider_ids)
    except TypeError:
        # Older sfmapi without the ``providers=`` kwarg on the registrar.
        register_backend("mapanything", backend_factory)


plugin = MapAnythingPlugin(
    manifest=manifest,
    backend_name="mapanything",
    backend_factory=backend_factory,
)


__all__ = [
    "TORCH_RUNTIME",
    "MapAnythingBackend",
    "MapAnythingPlugin",
    "PluginManifestDict",
    "ProviderManifestDict",
    "backend_factory",
    "get_plugin_manifest",
    "manifest",
    "plugin",
    "register",
]
