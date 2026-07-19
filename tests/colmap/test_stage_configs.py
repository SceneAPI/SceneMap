"""COLMAP stage-config vendor data, evicted from sfmapi core into the plugin.

Ports the coverage of core's former
``test_framework_colmap_descriptors_are_single_source_with_from_poses``:
the ``COLMAP_STAGE_CONFIGS`` table now lives here, and the three COLMAP
providers serve it (from_poses included, runtime-managed options filtered,
capability-gated) through their own ``list_backend_config_schemas``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sceneapi_map.colmap.cli.backend import ColmapCliBackend as CliBackend
from sceneapi_map.colmap.native.backend import ColmapCliBackend as NativeBackend
from sceneapi_map.colmap.pycolmap.backend import ColmapCliBackend as PycolmapBackend
from sceneapi_map.colmap.stage_configs import (
    COLMAP_STAGE_CONFIGS,
    RUNTIME_MANAGED_COLMAP_OPTIONS,
)

PROVIDER_BACKENDS = [
    pytest.param(CliBackend, id="cli"),
    pytest.param(NativeBackend, id="native"),
    pytest.param(PycolmapBackend, id="pycolmap"),
]


def _schema(command: str) -> dict[str, Any]:
    return {
        "command": command,
        "available": True,
        "schema_source": "test",
        "option_count": 2,
        "options": [
            # runtime-managed -> must be filtered out of the option schema.
            {"name": "database_path", "schema": {"type": "string"}},
            {"name": "SiftMatching.max_ratio", "schema": {"type": "number"}},
        ],
    }


def test_table_carries_from_poses_and_runtime_managed_set() -> None:
    config_ids = {row[0] for row in COLMAP_STAGE_CONFIGS}
    assert "colmap.pairs.from_poses" in config_ids
    from_poses = next(row for row in COLMAP_STAGE_CONFIGS if row[0] == "colmap.pairs.from_poses")
    _config_id, stage, capability, provider, command = from_poses
    assert (stage, capability, provider) == ("pairs", "pairs.from_poses", "colmap")
    # from_poses reuses the spatial_matcher option surface (same command).
    assert command == "spatial_matcher"
    # the runtime-managed set carries the COLMAP path/logging flags core used to filter.
    assert {"database_path", "image_path", "log_level"} <= RUNTIME_MANAGED_COLMAP_OPTIONS


@pytest.mark.parametrize("backend_cls", PROVIDER_BACKENDS)
def test_provider_serves_from_poses_schema_capability_gated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, backend_cls
) -> None:
    fake = tmp_path / "colmap"
    fake.write_text("", encoding="utf-8")
    backend = backend_cls(executable=fake)
    monkeypatch.setattr(backend, "colmap_command_schema", _schema)
    # advertise from_poses but NOT map.global, to prove capability gating.
    monkeypatch.setattr(backend, "capabilities", lambda: {"pairs.from_poses"})

    rows = backend.list_backend_config_schemas(include_schemas=True)
    by_id = {row["config_id"]: row for row in rows}

    assert "colmap.pairs.from_poses" in by_id
    from_poses = by_id["colmap.pairs.from_poses"]
    assert from_poses["capability"] == "pairs.from_poses"
    # description interpolates the capability (no literal "{capability}" leak).
    assert "{capability}" not in (from_poses.get("description") or "")
    props = from_poses["option_schema"]["properties"]
    assert "SiftMatching.max_ratio" in props
    assert "database_path" not in props  # runtime-managed, filtered out
    # capability-gated: map.global isn't advertised, so it isn't served.
    assert "colmap.mapping.global" not in by_id


@pytest.mark.parametrize("backend_cls", PROVIDER_BACKENDS)
def test_served_config_schemas_satisfy_core_contract_checker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, backend_cls
) -> None:
    backend_config = pytest.importorskip("sceneapi.server.adapters.backend_config")
    fake = tmp_path / "colmap"
    fake.write_text("", encoding="utf-8")
    backend = backend_cls(executable=fake)
    monkeypatch.setattr(backend, "colmap_command_schema", _schema)
    monkeypatch.setattr(backend, "capabilities", lambda: {"pairs.from_poses", "map.global"})

    assert backend_config.backend_config_contract_violations(backend) == []
