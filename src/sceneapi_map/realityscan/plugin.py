"""sfmapi plugin entry point for the RealityScan CLI backend."""

from __future__ import annotations

from collections.abc import Callable
from typing import NotRequired, TypedDict

from .backend import RealityScanCliBackend


class ProviderManifest(TypedDict):
    provider_id: str
    display_name: str
    capabilities: list[str]
    backend_actions: list[str]
    priority_hint: int


class UvRuntime(TypedDict):
    source: str
    url: str
    ref: str
    package: str


class ExternalToolRuntime(TypedDict):
    executable_names: list[str]
    env_vars: list[str]
    version_args: list[str]


class RuntimeModes(TypedDict):
    uv: UvRuntime
    external_tool: ExternalToolRuntime


class LicenseInfo(TypedDict):
    name: str


class UpstreamProject(TypedDict):
    name: str
    url: str
    license: str


class Compatibility(TypedDict):
    sfmapi: str
    python: str
    os: list[str]
    tool_versions: dict[str, str]


class Conformance(TypedDict):
    status: str
    suite: str


class PluginManifest(TypedDict):
    schema_version: int
    plugin_id: str
    display_name: str
    description: str
    package_name: str
    github_url: str
    entry_points: list[str]
    providers: list[ProviderManifest]
    runtime_modes: RuntimeModes
    capabilities: list[str]
    backend_actions: list[str]
    config_schemas: list[str]
    artifact_contracts: list[str]
    licenses: list[LicenseInfo]
    upstream_projects: list[UpstreamProject]
    compatibility: Compatibility
    conformance: Conformance
    trust_tier: NotRequired[str]


PLUGIN_ID = "realityscan_cli"
PROVIDER_ID = "realityscan_cli"
BACKEND_ACTIONS = ["realityscan.*"]
# RealityScan is an action-only backend: the proprietary project-based CLI has
# no portable sfmapi stage shape, so it exposes nothing through the portable
# capability vocabulary. ``RealityScanCliBackend.capabilities()`` returns an
# empty set and every portable stage method raises ``CapabilityUnavailableError``.
CAPABILITIES: list[str] = []

manifest: PluginManifest = {
    "schema_version": 1,
    "plugin_id": PLUGIN_ID,
    "display_name": "RealityScan CLI",
    "description": (
        "Backend plugin for installed RealityScan or RealityCapture command-line tools."
    ),
    "package_name": "sceneapi-map",
    "github_url": "https://github.com/SceneAPI/SceneMap.git",
    "entry_points": ["sceneapi_map.realityscan.plugin:plugin"],
    "providers": [
        {
            "provider_id": PROVIDER_ID,
            "display_name": "RealityScan CLI",
            "capabilities": CAPABILITIES,
            "backend_actions": BACKEND_ACTIONS,
            "priority_hint": 50,
        }
    ],
    "runtime_modes": {
        "uv": {
            "source": "git",
            "url": "https://github.com/SceneAPI/SceneMap.git",
            "ref": "main",
            "package": "sceneapi-map",
        },
        "external_tool": {
            "executable_names": ["RealityScan.exe", "RealityCapture.exe"],
            "env_vars": [
                "SFMAPI_RC_EXECUTABLE",
                "SFMAPI_REALITYCAPTURE_EXECUTABLE",
                "SFMAPI_REALITYSCAN_EXECUTABLE",
                "REALITYSCAN_EXE",
                "REALITYCAPTURE_EXE",
            ],
            "version_args": ["-version"],
        },
    },
    "capabilities": CAPABILITIES,
    "backend_actions": BACKEND_ACTIONS,
    # Action-only backend: ``capabilities`` and ``config_schemas`` stay empty
    # (no portable stage shape, no ``list_backend_config_schemas`` method).
    # ``RealityScanCliBackend.list_backend_artifact_contracts`` does describe
    # the registration / sparse-point-cloud exports its alignment verbs
    # produce; those contracts carry ``capability=None`` (action-only).
    "config_schemas": [],
    "artifact_contracts": ["realityscan.registration", "realityscan.sparse_point_cloud"],
    "licenses": [{"name": "Apache-2.0"}],
    "upstream_projects": [
        {
            "name": "RealityScan",
            "url": "https://www.capturingreality.com/realityscan",
            "license": "Proprietary",
        }
    ],
    "compatibility": {
        "sfmapi": ">=0.0.1",
        "python": ">=3.12,<3.13",
        "os": ["windows"],
        "tool_versions": {"RealityScan": "current", "RealityCapture": "current"},
    },
    "conformance": {"status": "not_run", "suite": "sfmapi-bench"},
    "trust_tier": "official",
}

backend_name = PROVIDER_ID
backend_factory = RealityScanCliBackend


def get_plugin_manifest() -> PluginManifest:
    return manifest


def _register_with_provider_aliases(register_backend: Callable[..., None]) -> None:
    provider_ids = [str(provider["provider_id"]) for provider in manifest["providers"]]
    try:
        register_backend(PROVIDER_ID, RealityScanCliBackend, providers=provider_ids)
    except TypeError:
        # Older sfmapi without ``providers=`` kwarg on the registrar.
        register_backend(PROVIDER_ID, RealityScanCliBackend)


def register_backend(register_backend: Callable[..., None]) -> None:
    _register_with_provider_aliases(register_backend)


def register(register_backend: Callable[..., None]) -> None:
    _register_with_provider_aliases(register_backend)


# `RealityScanCliPlugin` is the per-plugin name this module has always
# exported; alias the canonical sceneapi.backends.Plugin so downstream
# consumers (and the existing __all__) keep working. RealityScan's
# registration registers the same backend under multiple alias provider
# ids, so we pass the existing alias helper as the canonical Plugin's
# `register_hook` to keep behavior identical.
from sceneapi.backends import Plugin as RealityScanCliPlugin  # noqa: E402

plugin = RealityScanCliPlugin(
    manifest=manifest,
    backend_name=PROVIDER_ID,
    backend_factory=RealityScanCliBackend,
    register_hook=_register_with_provider_aliases,
)


__all__ = [
    "BACKEND_ACTIONS",
    "CAPABILITIES",
    "PLUGIN_ID",
    "PROVIDER_ID",
    "RealityScanCliPlugin",
    "backend_factory",
    "backend_name",
    "get_plugin_manifest",
    "manifest",
    "plugin",
    "register",
    "register_backend",
]
