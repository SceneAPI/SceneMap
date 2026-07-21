"""Install-time runtime provisioning for the unified COLMAP plugins.

Merges the three per-repo ``provisioning.py`` modules. The shared
machinery (GitHub release download on Windows, CMake source build
elsewhere, pycolmap wheel install) existed in near-identical copies;
the per-provider ``provision_*`` entry points preserve each source
repo's exact behavior:

- :func:`provision_native` (ex ``sfmapi_colmap``): COLMAP executable
  (env/PATH/cache probe, then release download or source build) plus
  the pycolmap wheel.
- :func:`provision_pycolmap` (ex ``sfmapi_pycolmap``): pycolmap wheel
  only.
- :func:`provision_cli` (ex ``sfmapi_colmap_cli``): COLMAP executable
  only, detected/registered through the cli provider's
  ``configure_colmap_environment``.

:func:`provision` is the ``sfm_hub`` package hook
(``<package>.provisioning:provision``); since this package ships every
provider it runs the superset (= native) path.

Unification note: the GitHub API User-Agent header is now
``scenemap`` for every provider (the sources sent their
own package names; cosmetic).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

RELEASE_API = "https://api.github.com/repos/colmap/colmap/releases/latest"
SOURCE_REPO = "https://github.com/colmap/colmap.git"
USER_AGENT = "scenemap"


def _cache_root() -> Path:
    override = os.environ.get("SFMAPI_PLUGIN_CACHE")
    if override:
        return Path(os.path.expandvars(override)).expanduser() / "colmap"
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "sfmapi" / "plugins" / "colmap"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "sfmapi" / "plugins" / "colmap"


def _step(name: str, action: str, status: str, **extra: object) -> dict[str, object]:
    return {"name": name, "action": action, "status": status, **extra}


def _existing_colmap() -> Path | None:
    env = os.environ.get("SFMAPI_COLMAP_EXECUTABLE")
    if env and Path(env).exists():
        return Path(env).resolve()
    found = shutil.which("colmap")
    if found:
        return Path(found).resolve()
    cache = _cache_root()
    names = ["colmap.exe", "colmap.bat", "colmap"] if os.name == "nt" else ["colmap"]
    candidates: list[Path] = []
    for name in names:
        candidates.extend([cache / "current" / name, cache / "current" / "bin" / name])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _pycolmap_installed() -> bool:
    return importlib.util.find_spec("pycolmap") is not None


def _install_pycolmap(steps: list[dict[str, object]], *, force: bool) -> None:
    if _pycolmap_installed() and not force:
        steps.append(_step("pycolmap_wheel", "use installed pycolmap wheel", "skipped"))
        return
    steps.append(_step("pycolmap_wheel", "uv pip install pycolmap>=4.0.4", "running"))
    subprocess.run(
        ["uv", "pip", "install", "pycolmap>=4.0.4"],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        text=True,
    )
    steps[-1]["status"] = "done"


def _request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _download(url: str, target: Path, *, digest: str | None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=180) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out)
    if digest and digest.startswith("sha256:"):
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        expected = digest.removeprefix("sha256:")
        if actual.lower() != expected.lower():
            raise RuntimeError(f"COLMAP release checksum mismatch: {actual} != {expected}")


def _select_release_asset(release: dict[str, Any]) -> dict[str, Any]:
    flavor = os.environ.get("SFMAPI_COLMAP_RELEASE_FLAVOR", "nocuda").casefold()
    assets = release.get("assets")
    if not isinstance(assets, list):
        assets = []
    for asset in assets:
        name = str(asset.get("name") or "").casefold()
        url = str(asset.get("browser_download_url") or "")
        tokens = set(re.split(r"[^a-z0-9]+", name))
        if name.endswith(".zip") and "windows" in tokens and flavor in tokens and url:
            return asset
    for asset in assets:
        name = str(asset.get("name") or "").casefold()
        url = str(asset.get("browser_download_url") or "")
        if name.endswith(".zip") and "windows" in name and url:
            return asset
    raise RuntimeError("latest COLMAP release has no Windows zip asset")


def _install_release(*, force: bool, steps: list[dict[str, object]]) -> Path:
    cache = _cache_root()
    current = cache / "current"
    existing = current / ("colmap.exe" if os.name == "nt" else "colmap")
    if existing.exists() and not force:
        steps.append(_step("colmap_release", "use cached COLMAP release", "skipped"))
        return existing

    release = _request_json(RELEASE_API)
    asset = _select_release_asset(release)
    archive = cache / "_downloads" / str(asset["name"])
    steps.append(_step("colmap_release", f"download {asset['browser_download_url']}", "running"))
    _download(str(asset["browser_download_url"]), archive, digest=asset.get("digest"))

    with tempfile.TemporaryDirectory(prefix="sfmapi_colmap_") as tmp:
        staging = Path(tmp)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(staging)
        found = sorted(staging.rglob("colmap.exe" if os.name == "nt" else "colmap"))
        if not found:
            raise RuntimeError(f"COLMAP archive {archive} did not contain colmap")
        source_dir = found[0].parent
        if current.exists():
            shutil.rmtree(current)
        shutil.copytree(source_dir, current)

    executable = current / found[0].name
    steps[-1]["status"] = "done"
    steps[-1]["path"] = str(executable)
    return executable


def _build_from_source(*, force: bool, steps: list[dict[str, object]]) -> Path:
    cache = _cache_root()
    source = cache / "source"
    build = cache / "build"
    executable_name = "colmap.exe" if os.name == "nt" else "colmap"
    candidates = [
        build / "src" / "exe" / "Release" / executable_name,
        build / "src" / "exe" / executable_name,
        build / "src" / "colmap" / "exe" / executable_name,
    ]
    for candidate in candidates:
        if candidate.exists() and not force:
            steps.append(_step("colmap_build", "use cached COLMAP build", "skipped"))
            return candidate

    cache.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        steps.append(_step("colmap_source", f"clone {SOURCE_REPO}", "running"))
        subprocess.run(
            ["git", "clone", "--recursive", SOURCE_REPO, str(source)],
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
        )
        steps[-1]["status"] = "done"
    steps.append(_step("colmap_build", "configure and build with CMake", "running"))
    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "--config", "Release", "--parallel"],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        text=True,
    )
    for candidate in candidates:
        if candidate.exists():
            steps[-1]["status"] = "done"
            steps[-1]["path"] = str(candidate)
            return candidate
    raise RuntimeError(f"COLMAP build completed but no {executable_name} was found")


def _install_colmap_executable(*, force: bool, steps: list[dict[str, object]]) -> Path:
    if os.name == "nt":
        return _install_release(force=force, steps=steps)
    return _build_from_source(force=force, steps=steps)


def _dry_run_colmap_step() -> dict[str, object]:
    action = "download latest COLMAP Windows release"
    if os.name != "nt":
        action = "clone COLMAP and build with CMake"
    return _step("colmap_runtime", action, "planned")


def provision_native(*, dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    """COLMAP executable + pycolmap wheel (ex ``sfmapi_colmap``)."""

    steps: list[dict[str, object]] = []
    if dry_run:
        return {
            "available": True,
            "provisioned": False,
            "steps": [
                _dry_run_colmap_step(),
                _step("pycolmap_wheel", "uv pip install pycolmap>=4.0.4", "planned"),
            ],
            "env": {},
            "warnings": [],
        }

    existing = _existing_colmap()
    if existing and not force:
        steps.append(_step("colmap_executable", "use configured COLMAP executable", "skipped"))
        _install_pycolmap(steps, force=force)
        return {
            "available": True,
            "provisioned": True,
            "steps": steps,
            "env": {"SFMAPI_COLMAP_EXECUTABLE": str(existing)},
            "warnings": [],
        }

    executable = _install_colmap_executable(force=force, steps=steps)
    _install_pycolmap(steps, force=force)
    os.environ["SFMAPI_COLMAP_EXECUTABLE"] = str(executable)
    return {
        "available": True,
        "provisioned": True,
        "steps": steps,
        "env": {"SFMAPI_COLMAP_EXECUTABLE": str(executable)},
        "warnings": [],
    }


def provision_pycolmap(*, dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    """pycolmap wheel only (ex ``sfmapi_pycolmap``)."""

    if dry_run:
        return {
            "available": True,
            "provisioned": False,
            "steps": [_step("pycolmap_wheel", "uv pip install pycolmap>=4.0.4", "planned")],
            "env": {},
            "warnings": [],
        }
    if _pycolmap_installed() and not force:
        return {
            "available": True,
            "provisioned": True,
            "steps": [_step("pycolmap_wheel", "use installed pycolmap wheel", "skipped")],
            "env": {},
            "warnings": [],
        }
    steps = [_step("pycolmap_wheel", "uv pip install pycolmap>=4.0.4", "running")]
    subprocess.run(
        ["uv", "pip", "install", "pycolmap>=4.0.4"],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        text=True,
    )
    steps[-1]["status"] = "done"
    return {
        "available": True,
        "provisioned": True,
        "steps": steps,
        "env": {},
        "warnings": [],
    }


def provision_cli(*, dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    """COLMAP executable only, via the cli provider (ex ``sfmapi_colmap_cli``)."""

    from .cli.backend import configure_colmap_environment

    steps: list[dict[str, object]] = []
    if dry_run:
        return {
            "available": True,
            "provisioned": False,
            "steps": [_dry_run_colmap_step()],
            "env": {},
            "warnings": [],
        }

    existing = configure_colmap_environment(validate=False)
    if existing and not force:
        return {
            "available": True,
            "provisioned": True,
            "steps": [_step("colmap_executable", "use configured COLMAP executable", "skipped")],
            "env": {"SFMAPI_COLMAP_EXECUTABLE": str(existing)},
            "warnings": [],
        }

    executable = _install_colmap_executable(force=force, steps=steps)
    os.environ["SFMAPI_COLMAP_EXECUTABLE"] = str(executable)
    configure_colmap_environment(executable)
    return {
        "available": True,
        "provisioned": True,
        "steps": steps,
        "env": {"SFMAPI_COLMAP_EXECUTABLE": str(executable)},
        "warnings": [],
    }


def provision(*, dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    """Package-level ``sfm_hub`` hook: provision for every provider.

    The union of the three providers' needs is exactly the native path
    (COLMAP executable + pycolmap wheel), so delegate to it.
    """

    return provision_native(dry_run=dry_run, force=force)
