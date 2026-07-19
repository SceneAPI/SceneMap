"""COLMAP stage-config vendor data — owned by the COLMAP plugin family.

The canonical COLMAP stage-config table (``COLMAP_STAGE_CONFIGS``) and the
set of runtime-managed COLMAP options (``RUNTIME_MANAGED_COLMAP_OPTIONS``)
live HERE, next to the backends that produce them, rather than in the
sfmapi core. Core owns the generic ``BackendConfigSchemaProvider``
discovery machinery; the COLMAP-specific rows are vendor data the three
COLMAP providers (colmap_cli / colmap_native / pycolmap) serve through
their own ``list_backend_config_schemas`` implementations.

This is the single source of truth for the three providers so their
served config-schema surfaces cannot drift (a previous split had lost the
``from_poses`` row from one plugin).
"""

from __future__ import annotations

#: ``(config_id, stage, capability, provider, command)`` for every COLMAP
#: stage option surface, keyed by the capability that gates it. A provider
#: serves the row only when it advertises ``capability``.
COLMAP_STAGE_CONFIGS: tuple[tuple[str, str, str, str, str], ...] = (
    ("colmap.features.sift", "features", "features.extract.sift", "colmap", "feature_extractor"),
    ("colmap.pairs.exhaustive", "pairs", "pairs.exhaustive", "colmap", "exhaustive_matcher"),
    ("colmap.pairs.sequential", "pairs", "pairs.sequential", "colmap", "sequential_matcher"),
    ("colmap.pairs.spatial", "pairs", "pairs.spatial", "colmap", "spatial_matcher"),
    ("colmap.pairs.vocabtree", "pairs", "pairs.vocabtree", "colmap", "vocab_tree_matcher"),
    ("colmap.pairs.explicit", "pairs", "pairs.explicit", "colmap", "matches_importer"),
    # from_poses selects pairs by camera-position proximity, reusing the
    # spatial_matcher option surface (same COLMAP command).
    ("colmap.pairs.from_poses", "pairs", "pairs.from_poses", "colmap", "spatial_matcher"),
    ("colmap.matcher.sift", "matcher", "matchers.nn-mutual", "colmap", "exhaustive_matcher"),
    ("colmap.verify", "verify", "matches.verify", "colmap", "geometric_verifier"),
    ("colmap.mapping.incremental", "mapping", "map.incremental", "colmap", "mapper"),
    ("colmap.mapping.global", "mapping", "map.global", "colmap", "global_mapper"),
    ("colmap.mapping.hierarchical", "mapping", "map.hierarchical", "colmap", "hierarchical_mapper"),
    ("colmap.ba.standard", "bundle_adjustment", "ba.standard", "colmap", "bundle_adjuster"),
)

#: COLMAP CLI options sfmapi supplies at runtime (paths, logging). They are
#: filtered out of the served ``backend_options`` schemas so clients cannot
#: set them.
RUNTIME_MANAGED_COLMAP_OPTIONS: frozenset[str] = frozenset(
    {
        "database_path",
        "image_path",
        "image_list_path",
        "input_path",
        "input_path1",
        "input_path2",
        "output_path",
        "workspace_path",
        "project_path",
        "match_list_path",
        "help",
        "log_level",
        "log_to_stderr",
        "log_color",
        "log_target",
    }
)


__all__ = [
    "COLMAP_STAGE_CONFIGS",
    "RUNTIME_MANAGED_COLMAP_OPTIONS",
]
