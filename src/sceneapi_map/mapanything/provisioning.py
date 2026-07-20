"""Install-time runtime provisioning for the MapAnything backend.

MapAnything's engine is heavy and git-only (the ``mapanything`` package is
not on PyPI) so, mirroring the InstantSfM provisioner, it is NOT a declared
pip dependency — it is installed here at provisioning time along with a CUDA
Torch runtime. Model weights are fetched lazily by ``from_pretrained`` on the
first inference; provisioning only PRE-fetches them when explicitly asked
(``SCENEAPI_MAPANYTHING_PREFETCH_WEIGHTS=1``) — tests never download weights.
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import subprocess
from typing import Any

from .backend import APACHE_WEIGHTS, DEFAULT_WEIGHTS, WEIGHTS_ENV_VAR, resolve_weights

MAPANYTHING_REPO = "https://github.com/facebookresearch/map-anything.git"
MAPANYTHING_REF = "main"
TORCH_DEFAULT_DEVICE = "cuda"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
TORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
TORCH_PACKAGES = ("torch", "torchvision")
# `pip install -e ".[all]"` from the upstream checkout pulls MapAnything's own
# runtime deps; the git spec below installs the package + its `all` extra.
MAPANYTHING_INSTALL_SPEC = f"mapanything[all] @ git+{MAPANYTHING_REPO}@{MAPANYTHING_REF}"
RUNTIME_READY_MODULES = ("torch", "mapanything")


def _step(name: str, action: str, status: str, **extra: object) -> dict[str, object]:
    return {"name": name, "action": action, "status": status, **extra}


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _tail(value: str, limit: int = 6000) -> str:
    return value[-limit:] if len(value) > limit else value


def _run_checked(
    args: list[str],
    steps: list[dict[str, object]],
    *,
    name: str,
    action: str | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    steps.append(_step(name, action or shlex.join(args), "running"))
    try:
        completed = subprocess.run(
            args,
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        steps[-1]["status"] = "failed"
        steps[-1]["returncode"] = exc.returncode
        if exc.stderr:
            steps[-1]["stderr_tail"] = _tail(exc.stderr)
        detail = _tail(exc.stderr or exc.stdout or str(exc))
        raise RuntimeError(f"{name} failed: {detail}") from exc
    steps[-1]["status"] = "done"
    if completed.stdout:
        steps[-1]["stdout_tail"] = _tail(completed.stdout)
    return completed


def _torch_device() -> str:
    device = os.environ.get("TORCH_DEVICE", TORCH_DEFAULT_DEVICE).strip().lower()
    if device not in {"cpu", "cuda"}:
        raise RuntimeError("TORCH_DEVICE must be either 'cpu' or 'cuda'")
    return device


def _torch_packages() -> list[str]:
    return shlex.split(os.environ.get("TORCH_PACKAGES", "")) or list(TORCH_PACKAGES)


def _torch_index_url(device: str) -> str:
    if device == "cpu":
        return os.environ.get("TORCH_CPU_INDEX_URL", TORCH_CPU_INDEX_URL)
    return os.environ.get("TORCH_INDEX_URL", TORCH_INDEX_URL)


def _ensure_torch(steps: list[dict[str, object]], *, force: bool) -> None:
    if _installed("torch") and not force:
        steps.append(_step("torch_runtime", "use installed Torch", "skipped"))
        return
    device = _torch_device()
    index_url = _torch_index_url(device)
    packages = _torch_packages()
    _run_checked(
        ["uv", "pip", "install", "--reinstall", "--index-url", index_url, *packages],
        steps,
        name="torch_runtime",
        action=f"uv pip install --index-url {index_url} {' '.join(packages)}",
    )


def _ensure_mapanything(steps: list[dict[str, object]], *, force: bool) -> None:
    if _env_flag("SCENEAPI_MAPANYTHING_SKIP_ENGINE"):
        steps.append(_step("mapanything_engine", "skip MapAnything engine install", "skipped"))
        return
    if _installed("mapanything") and not force:
        steps.append(_step("mapanything_engine", "use installed mapanything", "skipped"))
        return
    spec = os.environ.get("SCENEAPI_MAPANYTHING_INSTALL_SPEC", MAPANYTHING_INSTALL_SPEC)
    _run_checked(
        ["uv", "pip", "install", spec],
        steps,
        name="mapanything_engine",
        action=f"uv pip install {spec}",
    )


def _prefetch_weights(steps: list[dict[str, object]], warnings: list[str]) -> None:
    weights = resolve_weights(None)
    if not _env_flag("SCENEAPI_MAPANYTHING_PREFETCH_WEIGHTS"):
        warnings.append(
            "MapAnything weights are fetched lazily on first inference. Set "
            "SCENEAPI_MAPANYTHING_PREFETCH_WEIGHTS=1 to download them now. The "
            f"default weights are {DEFAULT_WEIGHTS} (Apache-2.0); the CC-BY-NC-4.0 "
            f"weights are opt-in via {WEIGHTS_ENV_VAR}."
        )
        steps.append(
            _step(
                "mapanything_weights",
                "defer weights download to first inference",
                "skipped",
                weights=weights,
            )
        )
        return
    # from_pretrained caches into HF_HOME; run it in a subprocess so a failed
    # (network) download does not import torch into this process.
    code = f"from mapanything.models import MapAnything; MapAnything.from_pretrained({weights!r})"
    _run_checked(
        ["python", "-c", code],
        steps,
        name="mapanything_weights",
        action=f"prefetch weights {weights}",
    )


def _runtime_ready() -> bool:
    return all(_installed(module) for module in RUNTIME_READY_MODULES)


def provision(*, dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    """Install the CUDA Torch runtime + the git-only MapAnything engine.

    ``dry_run`` returns the plan without side effects. Weights are only
    pre-downloaded when ``SCENEAPI_MAPANYTHING_PREFETCH_WEIGHTS=1``.
    """
    if dry_run:
        return {
            "available": True,
            "provisioned": False,
            "steps": [
                _step(
                    "torch_runtime",
                    "uv pip install --index-url <torch wheel index> torch torchvision",
                    "planned",
                ),
                _step(
                    "mapanything_engine",
                    f"uv pip install {MAPANYTHING_INSTALL_SPEC}",
                    "planned",
                ),
                _step(
                    "mapanything_weights",
                    f"defer/optionally prefetch weights ({APACHE_WEIGHTS} by default)",
                    "planned",
                ),
            ],
            "env": {},
            "warnings": [],
        }

    steps: list[dict[str, object]] = []
    warnings: list[str] = []
    _ensure_torch(steps, force=force)
    _ensure_mapanything(steps, force=force)
    _prefetch_weights(steps, warnings)
    return {
        "available": True,
        "provisioned": _runtime_ready(),
        "steps": steps,
        "env": {
            "TORCH_DEVICE": _torch_device(),
            WEIGHTS_ENV_VAR: os.environ.get(WEIGHTS_ENV_VAR, DEFAULT_WEIGHTS),
        },
        "warnings": warnings,
        "metadata": {
            "upstream_repo": MAPANYTHING_REPO,
            "default_weights": DEFAULT_WEIGHTS,
            "weights_env": WEIGHTS_ENV_VAR,
        },
    }


__all__ = ["provision"]
