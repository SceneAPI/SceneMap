"""sfmapi plugin entry point for the COLMAP CLI backend."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# `ColmapCliPlugin` is the per-plugin name this module has always
# exported; alias the canonical sceneapi.backends.Plugin so downstream
# consumers (and the existing __all__) keep working.
from sceneapi.backends import Plugin as ColmapCliPlugin

from .backend import ColmapCliBackend

Manifest = dict[str, Any]
# Accepts ``providers=[...]`` on sfmapi >= the registrar-providers change.
RegisterBackend = Callable[..., None]


MANIFEST: Manifest = {
    "schema_version": 1,
    "plugin_id": "colmap_cli",
    "display_name": "COLMAP CLI",
    "description": "Backend plugin that drives an installed COLMAP executable through sfmapi.",
    "package_name": "scenemap",
    "github_url": "https://github.com/SceneAPI/SceneMap.git",
    "entry_points": ["scenemap.colmap.cli.plugin:plugin"],
    "providers": [
        {
            "provider_id": "colmap_cli",
            "display_name": "COLMAP CLI",
            "description": "COLMAP command-line provider using an external colmap executable.",
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
            "priority_hint": 20,
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
        "external_tool": {
            "executable_names": ["colmap", "colmap.exe"],
            "env_vars": ["SFMAPI_COLMAP_EXECUTABLE", "COLMAP_EXE", "SFMAPI_COLMAP_EXE"],
            "version_args": ["help"],
        },
    },
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
    "config_schemas": ["colmap.*"],
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
        "os": ["windows", "linux", "macos"],
        "tool_versions": {"colmap": ">=3.9"},
    },
    "conformance": {"status": "not_run", "suite": "sfmapi-bench"},
    "trust_tier": "official",
}


plugin = ColmapCliPlugin(
    manifest=MANIFEST,
    backend_name="colmap_cli",
    backend_factory=ColmapCliBackend,
)


__all__ = ["MANIFEST", "ColmapCliPlugin", "plugin"]
