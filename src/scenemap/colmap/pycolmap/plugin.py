"""sfmapi plugin hub entry point for the PyCOLMAP backend."""

from __future__ import annotations

from typing import Any, Protocol, TypedDict

# `PycolmapPlugin` is the per-plugin name this module has always
# exported; alias the canonical sceneapi.backends.Plugin so downstream
# consumers (and the existing __all__) keep working.
from sceneapi.backends import Plugin as PycolmapPlugin

from ..pycolmap_backend import PycolmapBackend


class ProviderManifest(TypedDict):
    provider_id: str
    display_name: str
    description: str
    capabilities: list[str]
    backend_actions: list[str]
    priority_hint: int


class PluginManifest(TypedDict):
    schema_version: int
    plugin_id: str
    display_name: str
    description: str
    package_name: str
    github_url: str
    entry_points: list[str]
    providers: list[ProviderManifest]
    runtime_modes: dict[str, dict[str, Any]]
    capabilities: list[str]
    backend_actions: list[str]
    config_schemas: list[str]
    artifact_contracts: list[str]
    licenses: list[dict[str, str]]
    upstream_projects: list[dict[str, str]]
    compatibility: dict[str, Any]
    conformance: dict[str, str]
    trust_tier: str


class RegisterBackend(Protocol):
    # ``providers=[...]`` is the new pattern on sfmapi after the
    # registrar-providers change; we declare both arities so callers
    # using either shape type-check cleanly.
    def __call__(
        self,
        name: str,
        factory: type[PycolmapBackend],
        *,
        providers: list[str] | None = None,
    ) -> None: ...


# Full set of portable capabilities ``PycolmapBackend.capabilities()``
# can return, ignoring runtime PATH/import gating:
#   - always-on pycolmap caps (incl. the in-process surfaces
#     localize.from_memory / geometry.two_view / georegister.gps and
#     pose_priors.mapping consumed by incremental_mapping);
#   - the CUDA-gated ``backend.actions`` flag;
#   - the COLMAP-CLI-gated extras (pairs.explicit, map.hierarchical,
#     relocalize.images, recon.merge, export.nvm, georegister.sim3,
#     pgo.optimize, image.undistort, index.vocab_tree, rigs.configure).
# ``compute.in_memory`` is intentionally absent: pycolmap runs
# in-process but still materializes a COLMAP database + sparse model
# on disk.
# Keep sorted.
CAPABILITIES = [
    "ba.standard",
    "backend.actions",
    "export.colmap_bin",
    "export.colmap_text",
    "export.nvm",
    "export.ply",
    "features.extract.sift",
    "geometry.two_view",
    "georegister.gps",
    "georegister.sim3",
    "image.undistort",
    "index.vocab_tree",
    "localize.from_memory",
    "map.global",
    "map.hierarchical",
    "map.incremental",
    "matchers.nn-mutual",
    "matchers.nn-ratio",
    "matches.verify",
    "pairs.exhaustive",
    "pairs.explicit",
    "pairs.from_poses",
    "pairs.sequential",
    "pairs.spatial",
    "pairs.vocabtree",
    "pgo.optimize",
    "pose_priors.mapping",
    "recon.merge",
    "relocalize.images",
    "rigs.configure",
    "triangulate.retri",
]

PLUGIN_MANIFEST: PluginManifest = {
    "schema_version": 1,
    "plugin_id": "pycolmap",
    "display_name": "pycolmap",
    "description": "Python backend plugin using pycolmap bindings for in-process COLMAP workflows.",
    "package_name": "scenemap",
    "github_url": "https://github.com/SceneAPI/SceneMap.git",
    "entry_points": ["scenemap.colmap.pycolmap.plugin:plugin"],
    "providers": [
        {
            "provider_id": "colmap_pycolmap",
            "display_name": "pycolmap",
            "description": "In-process pycolmap provider.",
            "capabilities": CAPABILITIES,
            # ``ColmapCliBackend`` emits ``colmap.*`` action ids (the
            # actions wrap the upstream COLMAP CLI), so the namespace
            # the manifest advertises must match what is emitted.
            "backend_actions": ["colmap.*"],
            "priority_hint": 30,
        }
    ],
    "runtime_modes": {
        "uv": {
            "source": "git",
            "url": "https://github.com/SceneAPI/SceneMap.git",
            "ref": "main",
            "package": "scenemap",
        },
        "docker": None,
    },
    "capabilities": CAPABILITIES,
    # ``list_backend_actions`` / ``list_backend_config_schemas`` (inherited
    # from ``ColmapCliBackend``) emit ``colmap.*`` ids because they wrap the
    # upstream COLMAP CLI — the manifest namespaces match the emitted ids.
    "backend_actions": ["colmap.*"],
    "config_schemas": ["colmap.*"],
    # Exactly the ``contract_id``s emitted by
    # ``ColmapCliBackend.list_backend_artifact_contracts``.
    "artifact_contracts": ["colmap.database", "colmap.sparse_model"],
    "licenses": [{"name": "Apache-2.0"}],
    "upstream_projects": [
        {
            "name": "pycolmap",
            "url": "https://github.com/colmap/pycolmap",
            "license": "BSD-3-Clause",
        },
        {
            "name": "COLMAP",
            "url": "https://github.com/colmap/colmap",
            "license": "BSD-3-Clause",
        },
    ],
    "compatibility": {
        "sfmapi": ">=0.0.1",
        "python": ">=3.12,<3.13",
        "os": ["windows", "linux", "macos"],
        "tool_versions": {"pycolmap": ">=3.11"},
    },
    "conformance": {"status": "not_run", "suite": "sfmapi-bench"},
    "trust_tier": "official",
}


plugin = PycolmapPlugin(
    manifest=PLUGIN_MANIFEST,
    backend_name="colmap_pycolmap",
    backend_factory=PycolmapBackend,
)


def get_plugin_manifest() -> PluginManifest:
    return plugin.get_plugin_manifest()


def register(register_backend: RegisterBackend) -> None:
    plugin.register(register_backend)


__all__ = [
    "PLUGIN_MANIFEST",
    "PycolmapPlugin",
    "get_plugin_manifest",
    "plugin",
    "register",
]
