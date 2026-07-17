from __future__ import annotations

import os
import tomllib
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from sceneapi_map.realityscan.backend import RealityScanCliBackend
from sceneapi_map.realityscan.plugin import PLUGIN_ID, PROVIDER_ID, manifest, plugin


def test_plugin_manifest_validates_against_sfmapi_contract() -> None:
    sfm_hub_models = pytest.importorskip("sfm_hub.models")

    validated = sfm_hub_models.PluginManifest.model_validate(manifest)

    assert validated.plugin_id == "realityscan_cli"
    assert validated.provider_ids() == ["realityscan_cli"]
    assert validated.entry_points == ["sceneapi_map.realityscan.plugin:plugin"]
    assert validated.runtime_modes.external_tool is not None
    assert validated.runtime_modes.external_tool.executable_names == [
        "RealityScan.exe",
        "RealityCapture.exe",
    ]
    assert validated.runtime_modes.external_tool.env_vars == [
        "SFMAPI_RC_EXECUTABLE",
        "SFMAPI_REALITYCAPTURE_EXECUTABLE",
        "SFMAPI_REALITYSCAN_EXECUTABLE",
        "REALITYSCAN_EXE",
        "REALITYCAPTURE_EXE",
    ]


def test_manifest_does_not_embed_local_tool_paths() -> None:
    external_tool = manifest["runtime_modes"]["external_tool"]

    for executable_name in external_tool["executable_names"]:
        assert not Path(executable_name).is_absolute()
        assert "\\" not in executable_name
        assert "/" not in executable_name
        assert os.pathsep not in executable_name


def test_pyproject_declares_importable_sfmapi_backend_entry_point() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    entry_points = pyproject["project"]["entry-points"]["sceneapi.backends"]

    # The unified pyproject now declares all six entry points; this
    # provider's row must keep its historical name (the source repo
    # asserted the whole single-entry table).
    assert entry_points[PLUGIN_ID] == "sceneapi_map.realityscan.plugin:plugin"

    entry_point = EntryPoint(
        name=PLUGIN_ID,
        value=entry_points[PLUGIN_ID],
        group="sceneapi.backends",
    )
    assert entry_point.load() is plugin


def test_plugin_registers_realityscan_backend_factory() -> None:
    registered: dict[str, object] = {}

    plugin.register(lambda name, factory: registered.setdefault(name, factory))

    assert PROVIDER_ID == "realityscan_cli"
    assert registered == {PROVIDER_ID: RealityScanCliBackend}
    assert isinstance(registered[PROVIDER_ID](), RealityScanCliBackend)
