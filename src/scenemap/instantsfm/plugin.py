from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

# `SfmapiBackendPlugin` is the per-plugin name this module has
# always exported; alias the canonical sceneapi.backends.Plugin so
# downstream consumers (and the existing __all__) keep working.
from sceneapi.backends import Plugin as SfmapiBackendPlugin

from .backend import InstantSfMBackend


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


TORCH_RUNTIME = {
    "policy": "required",
    "device": "cuda",
    "index_url": "https://download.pytorch.org/whl/cu128",
    "cpu_index_url": "https://download.pytorch.org/whl/cpu",
    "packages": ["torch", "torchvision", "torchaudio"],
    "install_env": {
        "TORCH_DEVICE": "cuda",
        "TORCH_INDEX_URL": "https://download.pytorch.org/whl/cu128",
        "TORCH_CPU_INDEX_URL": "https://download.pytorch.org/whl/cpu",
        "TORCH_PACKAGES": "torch torchvision torchaudio",
    },
}


manifest: PluginManifestDict = {
    "schema_version": 1,
    "plugin_id": "instantsfm",
    "display_name": "InstantSfM",
    "description": (
        "Wrapper + SDK material is Apache-2.0. Upstream InstantSfM "
        "(cre185/InstantSfM) is CC-BY-NC-4.0; that non-commercial term "
        "is upstream's and binds whoever runs InstantSfM, not added by "
        "sfmapi."
    ),
    "package_name": "scenemap",
    "github_url": "https://github.com/SceneAPI/SceneMap.git",
    "entry_points": ["scenemap.instantsfm.plugin:plugin"],
    "providers": [
        {
            "provider_id": "instantsfm",
            "display_name": "InstantSfM",
            # InstantSfM is a *global* SfM engine. It backs one portable
            # capability -- `map.global` -- via a path-staging adapter:
            # `InstantSfMBackend.run_mapping` stages a temp project root
            # whose `database.db` / `images` entries link to sfmapi's
            # independent paths, runs `scripts.sfm` against it, and reads
            # the COLMAP sparse model back out. Feature extraction stays
            # action-only: `scripts.feat` fuses extraction + matching
            # into one whole-project `GenerateDatabase` call (no separate
            # extract/pairs/match stages, and it refuses to run if the
            # database already exists), which does not map to a thin
            # portable-stage wrapper. Everything else is exposed via
            # `instantsfm.*` backend actions.
            "capabilities": ["map.global"],
            "backend_actions": ["instantsfm.*"],
            "priority_hint": 60,
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
        "container_service": {
            "protocol": "sfmapi-plugin-http-v1",
            "protocol_version": "1.0",
            "service": {"url_env": "SFMAPI_INSTANTSFM_SERVICE_URL"},
            "image": {
                "build": {
                    "source": "git",
                    "context": "https://github.com/SceneAPI/SceneMap.git",
                    "dockerfile": "Dockerfile",
                    "ref": "main",
                    "args": {
                        "TORCH_DEVICE": "cuda",
                        "TORCH_INDEX_URL": "https://download.pytorch.org/whl/cu128",
                        "TORCH_CPU_INDEX_URL": "https://download.pytorch.org/whl/cpu",
                        "TORCH_PACKAGES": "torch torchvision torchaudio",
                        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0;12.0",
                    },
                }
            },
            "object_store": {
                "url_env": "SFMAPI_PLUGIN_OBJECT_STORE_URL",
                "input_prefix": "instantsfm/input/",
                "output_prefix": "instantsfm/output/",
            },
            "cache": {
                "policy": "read_write",
                "scope": "plugin",
                "path": "/sfmapi/cache",
            },
            "provenance": {
                "image_digest_required": False,
                "source_revision": "main",
            },
            "healthcheck": {"path": "/healthz", "timeout_seconds": 5},
            "execution": {
                "path": "/execute",
                "timeout_seconds": 3600,
                "mounts": {
                    "input_path": "/sfmapi/input",
                    "output_path": "/sfmapi/output",
                    "work_path": "/sfmapi/work",
                    "log_path": "/sfmapi/logs",
                },
                "gpu": "required",
                "env": ["TORCH_HOME", "TORCH_DEVICE"],
                "secrets": [],
                "retry": {"max_attempts": 2, "backoff_seconds": 0},
                "shutdown_timeout_seconds": 10,
                "log_collection": "both",
                "artifact_collection": True,
            },
        },
    },
    # One portable capability -- `map.global` -- backed by the
    # `run_mapping` path-staging adapter. `config_schemas` now declares
    # `instantsfm.mapping.global`, served by `list_backend_config_schemas`
    # for the knobs `run_mapping` consumes; `artifact_contracts` stays `[]`
    # until backed by a real `list_backend_artifact_contracts` method. The
    # previously declared `features.extract.superpoint`, `pairs.retrieval`,
    # and `matchers.lightglue` had no backing code and have been removed
    # rather than faked; `map.incremental` was also the wrong name --
    # InstantSfM is a *global* SfM engine, so the portable stage it now
    # backs is `map.global`.
    "capabilities": ["map.global"],
    "backend_actions": ["instantsfm.*"],
    "config_schemas": ["instantsfm.mapping.global"],
    "artifact_contracts": [],
    "licenses": [{"name": "Apache-2.0"}],
    "upstream_projects": [
        {
            "name": "InstantSfM",
            "url": "https://github.com/cre185/InstantSfM",
            "license": "CC-BY-NC-4.0",
        }
    ],
    "compatibility": {
        "sfmapi": ">=0.0.1",
        "python": ">=3.12,<3.13",
        "os": ["windows", "linux"],
        "cuda": "required",
        "torch": TORCH_RUNTIME,
        "source_build": {
            "upstream_path": "third_party/instantsfm",
            "submodule": True,
            "install": "uv pip install --no-deps -e third_party/instantsfm",
            "dependency_overrides": {
                "scikit-sparse": (
                    "plugin-private SciPy-backed sksparse.cholmod shim "
                    "(PYTHONPATH-injected into worker subprocesses)"
                ),
                "pypose @ git+https://github.com/pypose/pypose.git@bae": "bae-kai",
            },
            "optional_groups": {
                "gaussian_splatting": {
                    "env": "SFMAPI_INSTANTSFM_INSTALL_GS=1",
                    "packages": [
                        "gsplat",
                        "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim",
                    ],
                }
            },
        },
    },
    "conformance": {"status": "not_run", "suite": "sfmapi-bench"},
    "trust_tier": "community",
}


def backend_factory() -> InstantSfMBackend:
    return InstantSfMBackend()


def get_plugin_manifest() -> PluginManifestDict:
    return manifest


def register(register_backend: Callable[..., None]) -> None:
    provider_ids = [str(provider["provider_id"]) for provider in manifest["providers"]]
    try:
        register_backend("instantsfm", backend_factory, providers=provider_ids)
    except TypeError:
        # Older sfmapi without ``providers=`` kwarg on the registrar.
        register_backend("instantsfm", backend_factory)


plugin = SfmapiBackendPlugin(
    manifest=manifest,
    backend_name="instantsfm",
    backend_factory=backend_factory,
)


__all__ = [
    "TORCH_RUNTIME",
    "PluginManifestDict",
    "SfmapiBackendPlugin",
    "backend_factory",
    "get_plugin_manifest",
    "manifest",
    "plugin",
    "register",
]
