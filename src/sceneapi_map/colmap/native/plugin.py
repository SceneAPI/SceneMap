"""sfmapi plugin entry point for COLMAP backend providers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..pycolmap_backend import PycolmapBackend
from .backend import ColmapCliBackend
from .cpp_inmemory_backend import CppInmemoryBackend
from .cpp_native_backend import CppNativeBackend

PLUGIN_ID = "colmap_native"

manifest: dict[str, Any] = {
    "schema_version": 1,
    "plugin_id": PLUGIN_ID,
    "display_name": "Native COLMAP backends",
    "description": (
        "COLMAP backend collection with CLI, pycolmap, native C++, and in-memory provider variants."
    ),
    "package_name": "sceneapi-map",
    "github_url": "https://github.com/SceneAPI/SceneMap.git",
    "entry_points": ["sceneapi_map.colmap.native.plugin:plugin"],
    "providers": [
        {
            "provider_id": "colmap_cli",
            "display_name": "COLMAP CLI",
            # Full union of ColmapCliBackend.capabilities() (ignores
            # runtime PATH gating, which can only ever shrink this set).
            "capabilities": [
                "ba.standard",
                "export.colmap_bin",
                "export.colmap_text",
                "export.nvm",
                "export.ply",
                "features.extract.sift",
                "georegister.gps",
                "georegister.sim3",
                "image.undistort",
                "index.vocab_tree",
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
            ],
            # Every COLMAP provider emits action ids under the
            # ``colmap.`` namespace (the descriptor methods in
            # backend.py hard-code that prefix; subclasses inherit it).
            "backend_actions": ["colmap.*"],
            "priority_hint": 25,
        },
        {
            "provider_id": "colmap_pycolmap",
            "display_name": "pycolmap",
            # Full union of PycolmapBackend.capabilities(): the
            # PyCOLMAP-native set (incl. PyCOLMAP-only
            # ``geometry.two_view``), the CUDA-gated ``backend.actions``,
            # and the CLI-fallback caps it gains when COLMAP is on PATH.
            "capabilities": [
                "backend.actions",
                "ba.standard",
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
            ],
            "backend_actions": ["colmap.*"],
            "priority_hint": 35,
        },
        {
            "provider_id": "colmap_cpp_native",
            "display_name": "COLMAP C++ native",
            # CppNativeBackend.capabilities() delegates to the full
            # ColmapCliBackend set, so it mirrors colmap_cli exactly
            # (PyCOLMAP-only ``geometry.two_view`` is NOT inherited).
            "capabilities": [
                "ba.standard",
                "export.colmap_bin",
                "export.colmap_text",
                "export.nvm",
                "export.ply",
                "features.extract.sift",
                "georegister.gps",
                "georegister.sim3",
                "image.undistort",
                "index.vocab_tree",
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
            ],
            "backend_actions": ["colmap.*"],
            "priority_hint": 40,
        },
        {
            "provider_id": "colmap_cpp_inmemory",
            "display_name": "COLMAP C++ in-memory",
            # CppInmemoryBackend.capabilities() returns ONLY these six
            # -- it does in-memory matching/verification (advertising
            # the ``compute.in_memory`` execution-mode flag) and has no
            # feature-extraction, mapping, or BA stage. It also emits
            # no backend actions (run_colmap_command raises).
            "capabilities": [
                "compute.in_memory",
                "matchers.nn-mutual",
                "matches.verify",
                "pairs.exhaustive",
                "pairs.explicit",
                "pairs.sequential",
            ],
            "backend_actions": [],
            "priority_hint": 45,
        },
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
            "executable_names": ["colmap", "colmap.exe"],
            "env_vars": ["SFMAPI_COLMAP_EXECUTABLE", "COLMAP_EXE", "SFMAPI_COLMAP_EXE"],
            "version_args": ["help"],
        },
    },
    # Sorted union of every provider's real capability set.
    "capabilities": [
        "backend.actions",
        "ba.standard",
        "compute.in_memory",
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
    ],
    # All four backends emit action ids under the single ``colmap.``
    # namespace; the ``pycolmap.*`` / ``colmap_cpp.*`` namespaces were
    # never produced by any implementation.
    "backend_actions": ["colmap.*"],
    "config_schemas": ["colmap.*"],
    # contract_ids emitted by ColmapCliBackend.list_backend_artifact_contracts()
    # (inherited by the pycolmap and C++ native providers; the C++
    # in-memory provider materializes no artifacts and emits none).
    "artifact_contracts": ["colmap.database", "colmap.sparse_model"],
    "licenses": [{"name": "Apache-2.0"}],
    "upstream_projects": [
        {
            "name": "COLMAP",
            "url": "https://github.com/colmap/colmap",
            "license": "BSD-3-Clause",
        }
    ],
    "compatibility": {
        "sfmapi": ">=0.0.1",
        "python": ">=3.12,<3.13",
        "os": ["windows", "linux"],
        "cuda": "optional",
        "tool_versions": {"colmap": ">=4.1"},
    },
    "conformance": {"status": "not_run", "suite": "sfmapi-bench"},
    "trust_tier": "official",
}

_BACKEND_FACTORIES: dict[str, Callable[[], Any]] = {
    "colmap_cli": ColmapCliBackend,
    "colmap_pycolmap": PycolmapBackend,
    "colmap_cpp_native": CppNativeBackend,
    "colmap_cpp_inmemory": CppInmemoryBackend,
}


def get_plugin_manifest() -> dict[str, Any]:
    """Return this package's sfmapi plugin manifest."""

    return manifest


def register(register_backend: Callable[..., None]) -> None:
    """Register every COLMAP provider factory with sfmapi.

    Each backend's name doubles as its sfm_hub provider id; declare
    ``providers=[name]`` explicitly so standalone deployments (which
    bypass sfm_hub discovery) still get provider aliasing without
    relying on the registry's name-fallback path.
    """

    for provider_id, factory in _BACKEND_FACTORIES.items():
        try:
            register_backend(provider_id, factory, providers=[provider_id])
        except TypeError:
            # Older sfmapi without ``providers=`` kwarg on the registrar.
            register_backend(provider_id, factory)


register_backend = register


# `ColmapNativePlugin` is the per-plugin name this module has always
# exported; alias the canonical sceneapi.backends.Plugin so downstream
# consumers (and the existing __all__) keep working. The colmap_native
# plugin registers four distinct backend factories under four provider
# ids in one manifest, so we pass the existing multi-provider register
# function as the canonical Plugin's `register_hook` to keep behavior
# identical. backend_name and backend_factory pick the colmap_cli row
# as the canonical 'primary' provider for introspection purposes (the
# manifest's first provider entry is colmap_cli).
from sceneapi.backends import Plugin as ColmapNativePlugin  # noqa: E402

plugin = ColmapNativePlugin(
    manifest=manifest,
    backend_name="colmap_cli",
    backend_factory=ColmapCliBackend,
    register_hook=register,
)

__all__ = [
    "PLUGIN_ID",
    "ColmapNativePlugin",
    "get_plugin_manifest",
    "manifest",
    "plugin",
    "register",
    "register_backend",
]
