"""Install-time runtime provisioning for the SphereSfM backend."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from .backend import resolve_spheresfm_executable

RELEASE_API = "https://api.github.com/repos/json87/SphereSfM/releases/latest"
SOURCE_REPO = "https://github.com/json87/SphereSfM.git"


def _cache_root() -> Path:
    override = os.environ.get("SFMAPI_PLUGIN_CACHE")
    if override:
        return Path(os.path.expandvars(override)).expanduser() / "spheresfm"
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "sfmapi" / "plugins" / "spheresfm"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "sfmapi" / "plugins" / "spheresfm"


def _step(name: str, action: str, status: str, **extra: object) -> dict[str, object]:
    return {"name": name, "action": action, "status": status, **extra}


def _request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "sceneapi-map"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _download(url: str, target: Path, *, digest: str | None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "sceneapi-map"})
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out)
    if digest and digest.startswith("sha256:"):
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        expected = digest.removeprefix("sha256:")
        if actual.lower() != expected.lower():
            raise RuntimeError(f"SphereSfM release checksum mismatch: {actual} != {expected}")


def _select_release_asset(release: dict[str, Any]) -> dict[str, Any]:
    assets = release.get("assets")
    if not isinstance(assets, list):
        assets = []
    for asset in assets:
        name = str(asset.get("name") or "")
        url = str(asset.get("browser_download_url") or "")
        if name.lower().endswith(".zip") and "spheresfm" in name.lower() and url:
            return asset
    raise RuntimeError("latest SphereSfM release has no downloadable zip asset")


def _install_release(*, force: bool, steps: list[dict[str, object]]) -> Path:
    cache = _cache_root()
    current = cache / "current"
    existing = current / ("colmap.exe" if os.name == "nt" else "colmap")
    if existing.exists() and not force:
        steps.append(_step("spheresfm_release", "use cached SphereSfM release", "skipped"))
        return existing

    release = _request_json(RELEASE_API)
    asset = _select_release_asset(release)
    archive = cache / "_downloads" / str(asset["name"])
    steps.append(
        _step(
            "spheresfm_release",
            f"download {asset['browser_download_url']}",
            "running",
        )
    )
    _download(str(asset["browser_download_url"]), archive, digest=asset.get("digest"))

    with tempfile.TemporaryDirectory(prefix="sfmapi_spheresfm_") as tmp:
        staging = Path(tmp)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(staging)
        found = sorted(staging.rglob("colmap.exe" if os.name == "nt" else "colmap"))
        if not found:
            raise RuntimeError(f"SphereSfM archive {archive} did not contain colmap")
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
        build / "src" / "exe" / "RelWithDebInfo" / executable_name,
    ]
    for candidate in candidates:
        if candidate.exists() and not force:
            steps.append(_step("spheresfm_build", "use cached SphereSfM build", "skipped"))
            return candidate

    cache.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        steps.append(_step("spheresfm_source", f"clone {SOURCE_REPO}", "running"))
        subprocess.run(
            ["git", "clone", "--recursive", SOURCE_REPO, str(source)],
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
        )
        steps[-1]["status"] = "done"
    elif force:
        steps.append(_step("spheresfm_source", "update existing SphereSfM checkout", "running"))
        subprocess.run(
            ["git", "-C", str(source), "pull", "--ff-only"],
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(source), "submodule", "update", "--init", "--recursive"],
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
        )
        steps[-1]["status"] = "done"

    steps.append(_step("spheresfm_build", "configure and build with CMake", "running"))
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
    raise RuntimeError(f"SphereSfM build completed but no {executable_name} was found")


def provision(*, dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    steps: list[dict[str, object]] = []
    existing = resolve_spheresfm_executable(None)
    if existing and not force:
        return {
            "available": True,
            "provisioned": True,
            "steps": [_step("spheresfm_executable", "use configured executable", "skipped")],
            "env": {"SFMAPI_SPHERESFM_EXECUTABLE": str(existing)},
            "warnings": [],
        }

    if dry_run:
        action = "download latest SphereSfM Windows release"
        if os.name != "nt":
            action = "clone SphereSfM and build with CMake"
        return {
            "available": True,
            "provisioned": False,
            "steps": [_step("spheresfm_runtime", action, "planned")],
            "env": {},
            "warnings": [],
        }

    if os.name == "nt":
        executable = _install_release(force=force, steps=steps)
    else:
        executable = _build_from_source(force=force, steps=steps)
    os.environ["SFMAPI_SPHERESFM_EXECUTABLE"] = str(executable)
    return {
        "available": True,
        "provisioned": True,
        "steps": steps,
        "env": {"SFMAPI_SPHERESFM_EXECUTABLE": str(executable)},
        "warnings": [],
    }
