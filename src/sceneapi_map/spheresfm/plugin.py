from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

# `SfmapiBackendPlugin` is the per-plugin name this module has always
# exported; alias the canonical sceneapi.backends.Plugin so downstream
# consumers (and the existing __all__) keep working.
from sceneapi.backends import Plugin as SfmapiBackendPlugin

from .backend import SphereSfMBackend


class ProviderManifestDict(TypedDict):
    provider_id: str
    display_name: str
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


manifest: PluginManifestDict = {
    "schema_version": 1,
    "plugin_id": "spheresfm",
    "display_name": "SphereSfM",
    "description": "Backend plugin for spherical Structure-from-Motion workflows.",
    "package_name": "sceneapi-map",
    "github_url": "https://github.com/SceneAPI/SceneMap.git",
    "entry_points": ["sceneapi_map.spheresfm.plugin:plugin"],
    "providers": [
        {
            "provider_id": "spheresfm",
            "display_name": "SphereSfM",
            # Portable capabilities backed by real wrapper methods on
            # SphereSfMBackend. SphereSfM is a COLMAP fork, so the full
            # sparse pipeline is wired: feature extraction, every
            # pair-selection matcher, incremental / hierarchical /
            # spherical mapping, standard + rig bundle adjustment,
            # re-triangulation, image registration, model merging,
            # export, and Sim(3) georegistration. SphereSfM's matchers
            # run geometric verification inline (no standalone verify
            # command), so ``matches.verify`` is deliberately not
            # advertised; nor is any image-only equirect transform.
            "capabilities": [
                "ba.rig",
                "ba.standard",
                "export.colmap_bin",
                "export.colmap_text",
                "export.nvm",
                "export.ply",
                "features.extract.sift",
                "georegister.sim3",
                "map.hierarchical",
                "map.incremental",
                "map.spherical",
                "pairs.exhaustive",
                "pairs.sequential",
                "pairs.spatial",
                "pairs.vocabtree",
                "projection.cubemap_rig",
                "recon.merge",
                "relocalize.images",
                "triangulate.retri",
            ],
            "backend_actions": ["spheresfm.*"],
            "priority_hint": 15,
        }
    ],
    "runtime_modes": {
        "uv": {
            "source": "git",
            "url": "https://github.com/SceneAPI/SceneMap.git",
            "ref": "main",
            "package": "sceneapi-map",
        },
        "docker": None,
        "external_tool": {
            "executable_names": ["spheresfm"],
            "env_vars": ["SFMAPI_SPHERESFM_EXECUTABLE", "SPHERESFM_EXE"],
            "version_args": ["help"],
        },
    },
    "capabilities": [
        "ba.rig",
        "ba.standard",
        "export.colmap_bin",
        "export.colmap_text",
        "export.nvm",
        "export.ply",
        "features.extract.sift",
        "georegister.sim3",
        "map.hierarchical",
        "map.incremental",
        "map.spherical",
        "pairs.exhaustive",
        "pairs.sequential",
        "pairs.spatial",
        "pairs.vocabtree",
        "projection.cubemap_rig",
        "recon.merge",
        "relocalize.images",
        "triangulate.retri",
    ],
    "backend_actions": ["spheresfm.*"],
    "config_schemas": ["spheresfm.*"],
    "artifact_contracts": [
        "spheresfm.matches.database",
        "spheresfm.reconstruction.spherical",
    ],
    "licenses": [{"name": "Apache-2.0"}],
    "upstream_projects": [
        {
            "name": "SphereSfM",
            "url": "https://github.com/json87/SphereSfM",
            "license": "BSD-3-Clause",
        }
    ],
    "compatibility": {
        "sfmapi": ">=0.0.1",
        "python": ">=3.12,<3.13",
        "os": ["windows", "linux"],
        "cuda": "optional",
    },
    "conformance": {"status": "not_run", "suite": "sfmapi-bench"},
    "trust_tier": "community",
}


def backend_factory() -> SphereSfMBackend:
    return SphereSfMBackend()


def get_plugin_manifest() -> PluginManifestDict:
    return manifest


def register(register_backend: Callable[..., None]) -> None:
    provider_ids = [str(provider["provider_id"]) for provider in manifest["providers"]]
    try:
        register_backend("spheresfm", backend_factory, providers=provider_ids)
    except TypeError:
        # Older sfmapi without ``providers=`` kwarg on the registrar.
        register_backend("spheresfm", backend_factory)


plugin = SfmapiBackendPlugin(
    manifest=manifest,
    backend_name="spheresfm",
    backend_factory=backend_factory,
)


__all__ = [
    "PluginManifestDict",
    "SfmapiBackendPlugin",
    "backend_factory",
    "get_plugin_manifest",
    "manifest",
    "plugin",
    "register",
]
