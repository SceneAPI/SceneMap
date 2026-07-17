"""Install-time runtime provisioning for the InstantSfM backend."""

from __future__ import annotations

import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from .backend import DEFAULT_INSTANTSFM_ROOT, REPO_ROOT, resolve_instantsfm_root

PLUGIN_REPO = "https://github.com/SceneAPI/SceneMap.git"
PLUGIN_REF = "main"
INSTANTSFM_SUBMODULE = Path("third_party") / "instantsfm"
TORCH_DEFAULT_DEVICE = "cuda"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
TORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
TORCH_PACKAGES = ("torch", "torchvision", "torchaudio")
DEFAULT_TORCH_CUDA_ARCH_LIST = "8.0;8.6;8.9;9.0;12.0"
CORE_RUNTIME_PACKAGES = (
    "numpy==1.26.4",
    "opencv-python==4.10.0.84",
    "pyyaml",
    "matplotlib",
    "scipy==1.13.0",
    "tqdm==4.66.5",
    "pyceres>=2.3,<3",
    "kornia==0.8.0",
    "viser==0.2.11",
    "imageio",
    "imageio-ffmpeg",
    "torchmetrics[image]",
    "tensorly",
    "nerfview",
    "tensorboard",
    "scikit-learn",
    "gradio",
    "plotly",
    "splines",
    "jaxtyping",
    "tyro",
    "typing-extensions",
)
WINDOWS_RUNTIME_PACKAGES = ("triton-windows",)
POSIX_RUNTIME_PACKAGES = ("triton",)
BAE_PACKAGES = ("bae-kai",)
GAUSSIAN_SPLATTING_PACKAGES = (
    "gsplat",
    "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim",
)
GAUSSIAN_SPLATTING_READY_MODULES = ("gsplat", "fused_ssim")
RUNTIME_READY_MODULES = (
    "cv2",
    "pyceres",
    "kornia",
    "viser",
    # scipy backs the plugin-private ``sksparse.cholmod`` shim
    # (sceneapi_map/instantsfm/_sksparse_shim), which is PYTHONPATH-injected into
    # worker subprocesses rather than installed as a top-level package.
    "scipy",
    "bae",
    "pypose",
)


def _cache_root() -> Path:
    override = os.environ.get("SFMAPI_PLUGIN_CACHE")
    if override:
        return Path(os.path.expandvars(override)).expanduser() / "instantsfm"
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "sfmapi" / "plugins" / "instantsfm"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "sfmapi" / "plugins" / "instantsfm"


def _step(name: str, action: str, status: str, **extra: object) -> dict[str, object]:
    return {"name": name, "action": action, "status": status, **extra}


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _install_gs_enabled() -> bool:
    return _env_flag("SFMAPI_INSTANTSFM_INSTALL_GS") and not _env_flag("SFMAPI_INSTANTSFM_SKIP_GS")


def _display_command(args: list[str]) -> str:
    return shlex.join(args)


def _tail(value: str, limit: int = 6000) -> str:
    return value[-limit:] if len(value) > limit else value


def _run_checked(
    args: list[str],
    steps: list[dict[str, object]],
    *,
    name: str,
    action: str | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    steps.append(_step(name, action or _display_command(args), "running"))
    try:
        completed = subprocess.run(
            args,
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            cwd=str(cwd) if cwd else None,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        steps[-1]["status"] = "failed"
        steps[-1]["returncode"] = exc.returncode
        if exc.stdout:
            steps[-1]["stdout_tail"] = _tail(exc.stdout)
        if exc.stderr:
            steps[-1]["stderr_tail"] = _tail(exc.stderr)
        detail = _tail(exc.stderr or exc.stdout or str(exc))
        raise RuntimeError(f"{name} failed: {detail}") from exc
    steps[-1]["status"] = "done"
    if completed.stdout:
        steps[-1]["stdout_tail"] = _tail(completed.stdout)
    if completed.stderr:
        steps[-1]["stderr_tail"] = _tail(completed.stderr)
    return completed


def _torch_device() -> str:
    device = os.environ.get("TORCH_DEVICE", TORCH_DEFAULT_DEVICE).strip().lower()
    if device not in {"cpu", "cuda"}:
        raise RuntimeError("TORCH_DEVICE must be either 'cpu' or 'cuda'")
    return device


def _torch_packages() -> list[str]:
    packages = shlex.split(os.environ.get("TORCH_PACKAGES", ""))
    return packages or list(TORCH_PACKAGES)


def _torch_index_url(device: str) -> str:
    if device == "cpu":
        return os.environ.get("TORCH_CPU_INDEX_URL", TORCH_CPU_INDEX_URL)
    return os.environ.get("TORCH_INDEX_URL", TORCH_INDEX_URL)


def _torch_state() -> tuple[str, str]:
    if not _installed("torch"):
        return "missing", "missing"

    import torch

    return str(torch.__version__), str(torch.version.cuda or "cpu")


def _torch_matches_device(device: str, state: tuple[str, str]) -> bool:
    _version, cuda = state
    if device == "cpu":
        return cuda == "cpu"
    return cuda not in {"cpu", "missing", ""}


def _ensure_torch_runtime(
    steps: list[dict[str, object]], *, force: bool, name: str = "torch_runtime"
) -> None:
    device = _torch_device()
    state = _torch_state()
    if not force and _torch_matches_device(device, state):
        version, cuda = state
        steps.append(
            _step(
                name,
                f"use installed {device} Torch",
                "skipped",
                torch_version=version,
                torch_cuda=cuda,
            )
        )
        return

    packages = _torch_packages()
    index_url = _torch_index_url(device)
    command = [
        ["uv", "pip", "install", "--reinstall", "--index-url", index_url, *packages],
    ][0]
    _run_checked(
        command,
        steps,
        name=name,
        action=f"uv pip install --reinstall --index-url {index_url} {' '.join(packages)}",
    )
    steps[-1]["device"] = device
    state = _torch_state()
    if not _torch_matches_device(device, state):
        version, cuda = state
        steps[-1]["status"] = "failed"
        steps[-1]["torch_version"] = version
        steps[-1]["torch_cuda"] = cuda
        raise RuntimeError(
            f"installed Torch does not match requested device {device!r}: "
            f"version={version}, cuda={cuda}"
        )
    version, cuda = state
    steps[-1]["status"] = "done"
    steps[-1]["torch_version"] = version
    steps[-1]["torch_cuda"] = cuda


def _source_root(*, force: bool, steps: list[dict[str, object]]) -> Path:
    raw_override = os.environ.get("SFMAPI_INSTANTSFM_ROOT")
    if raw_override:
        override = resolve_instantsfm_root(raw_override)
        if override is not None:
            steps.append(
                _step("instantsfm_source", f"use {override}", "skipped", root=str(override))
            )
            return override

    if (REPO_ROOT / ".gitmodules").exists():
        _ensure_instantsfm_submodule(REPO_ROOT, steps=steps, force=force)
        root = DEFAULT_INSTANTSFM_ROOT
    else:
        plugin_checkout = _ensure_plugin_checkout(steps=steps, force=force)
        root = plugin_checkout / INSTANTSFM_SUBMODULE

    if not _is_instantsfm_root(root):
        raise RuntimeError(
            "InstantSfM submodule is not populated. Run `git submodule update --init "
            "--recursive third_party/instantsfm` or rerun provisioning with network access."
        )
    return root.resolve()


def _is_instantsfm_root(root: Path) -> bool:
    return (root / "pyproject.toml").exists() and (root / "instantsfm").is_dir()


def _ensure_instantsfm_submodule(
    repo_root: Path, *, steps: list[dict[str, object]], force: bool
) -> None:
    root = repo_root / INSTANTSFM_SUBMODULE
    if _is_instantsfm_root(root) and not force:
        steps.append(
            _step(
                "instantsfm_submodule",
                f"use populated submodule {INSTANTSFM_SUBMODULE.as_posix()}",
                "skipped",
                root=str(root),
            )
        )
        return

    _run_checked(
        ["git", "submodule", "update", "--init", "--recursive", INSTANTSFM_SUBMODULE.as_posix()],
        steps,
        name="instantsfm_submodule",
        action=f"git submodule update --init --recursive {INSTANTSFM_SUBMODULE.as_posix()}",
        cwd=repo_root,
    )


def _ensure_plugin_checkout(*, steps: list[dict[str, object]], force: bool) -> Path:
    checkout = _cache_root() / "plugin_source"
    ref = os.environ.get("SFMAPI_INSTANTSFM_PLUGIN_REF", PLUGIN_REF)
    if (checkout / ".gitmodules").exists():
        if force:
            _run_checked(
                ["git", "fetch", "--tags", "--prune"],
                steps,
                name="plugin_source",
                action="git fetch --tags --prune cached SceneMap plugin checkout",
                cwd=checkout,
            )
            _run_checked(
                ["git", "checkout", ref],
                steps,
                name="plugin_source",
                action=f"git checkout {ref}",
                cwd=checkout,
            )
        else:
            steps.append(
                _step(
                    "plugin_source",
                    f"use cached SceneMap plugin checkout at {checkout}",
                    "skipped",
                    root=str(checkout),
                )
            )
        _ensure_instantsfm_submodule(checkout, steps=steps, force=force)
        return checkout

    if checkout.exists():
        raise RuntimeError(f"cache path {checkout} exists but is not a SceneMap plugin checkout")
    checkout.parent.mkdir(parents=True, exist_ok=True)
    _run_checked(
        ["git", "clone", "--recurse-submodules", PLUGIN_REPO, str(checkout)],
        steps,
        name="plugin_source",
        action=f"git clone --recurse-submodules {PLUGIN_REPO} {checkout}",
    )
    if ref:
        _run_checked(
            ["git", "checkout", ref],
            steps,
            name="plugin_source",
            action=f"git checkout {ref}",
            cwd=checkout,
        )
    _ensure_instantsfm_submodule(checkout, steps=steps, force=True)
    return checkout


def _replace_all(path: Path, replacements: dict[str, str]) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = original
    for old, new in replacements.items():
        updated = updated.replace(old, new)
    if not updated.endswith("\n"):
        updated += "\n"
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def _patch_instantsfm_source(root: Path, steps: list[dict[str, object]]) -> None:
    feature_handler = root / "instantsfm" / "controllers" / "feature_handler.py"
    scene_defs = root / "instantsfm" / "scene" / "defs.py"
    rotation_averaging = root / "instantsfm" / "processors" / "rotation_averaging.py"
    changed = _replace_all(
        feature_handler,
        {
            "--SiftExtraction.use_gpu": "--FeatureExtraction.use_gpu",
            "--SiftMatching.use_gpu": "--FeatureMatching.use_gpu",
            'print(f"Error during feature extraction: {e}")': (
                'raise RuntimeError(f"Error during feature extraction: {e}") from e'
            ),
            'print(f"Error during exhaustive matching: {e}")': (
                'raise RuntimeError(f"Error during exhaustive matching: {e}") from e'
            ),
        },
    )
    changed = (
        _replace_all(
            scene_defs,
            {
                "self.ids = np.full(num_tracks, -1, dtype=np.int32)": (
                    "self.ids = np.full(num_tracks, -1, dtype=np.int64)"
                ),
                "self.ids = np.zeros(self.num_tracks, dtype=np.int32)": (
                    "self.ids = np.zeros(self.num_tracks, dtype=np.int64)"
                ),
            },
        )
        or changed
    )
    changed = (
        _replace_all(
            rotation_averaging,
            {
                (
                    "        self.images = {image_id: image for image_id, image in "
                    "enumerate(images) if registered_mask[image_id]}\n"
                    "        self.image_pairs = {pair_key: pair for pair_key, pair in "
                    "view_graph.image_pairs.items() if pair.is_valid}"
                ): (
                    "        self.images = {\n"
                    "            image_id: image for image_id, image in enumerate(images) "
                    "if registered_mask[image_id]\n"
                    "        }\n"
                    "        registered_ids = set(self.images)\n"
                    "        self.image_pairs = {\n"
                    "            pair_key: pair\n"
                    "            for pair_key, pair in view_graph.image_pairs.items()\n"
                    "            if pair.is_valid\n"
                    "            and pair.image_id1 in registered_ids\n"
                    "            and pair.image_id2 in registered_ids\n"
                    "        }"
                ),
                "        if self.fixed_camera_id == -1:": (
                    "        if self.fixed_camera_id not in self.image_id2idx:"
                ),
            },
        )
        or changed
    )
    if not changed:
        steps.append(
            _step(
                "instantsfm_source_patches",
                "verify InstantSfM source compatibility patches",
                "skipped",
                root=str(root),
            )
        )
        return
    steps.append(
        _step(
            "instantsfm_source_patches",
            (
                "patch InstantSfM source for current COLMAP flags, hard failures, "
                "64-bit track IDs, and post-filter rotation averaging"
            ),
            "done",
            root=str(root),
        )
    )


def _installed_from_root(module: str, root: Path) -> bool:
    spec = importlib.util.find_spec(module)
    origin = getattr(spec, "origin", None) if spec else None
    locations = getattr(spec, "submodule_search_locations", None) if spec else None
    candidates = [origin] if origin else []
    if locations:
        candidates.extend(str(location) for location in locations)
    try:
        resolved_root = root.resolve()
        return any(
            Path(candidate).resolve().is_relative_to(resolved_root) for candidate in candidates
        )
    except OSError:
        return False


def _install_instantsfm_source(root: Path, steps: list[dict[str, object]], *, force: bool) -> None:
    if _installed_from_root("instantsfm", root) and not force:
        steps.append(
            _step(
                "instantsfm_source_install",
                f"use editable InstantSfM source install from {root}",
                "skipped",
                root=str(root),
            )
        )
        return
    _run_checked(
        ["uv", "pip", "install", "--no-deps", "-e", str(root)],
        steps,
        name="instantsfm_source_install",
        action=f"uv pip install --no-deps -e {root}",
    )


def _runtime_dependency_packages() -> list[str]:
    packages = list(CORE_RUNTIME_PACKAGES)
    packages.extend(WINDOWS_RUNTIME_PACKAGES if os.name == "nt" else POSIX_RUNTIME_PACKAGES)
    return packages


def _runtime_modules_ready() -> bool:
    modules = list(RUNTIME_READY_MODULES)
    if _install_gs_enabled():
        modules.extend(GAUSSIAN_SPLATTING_READY_MODULES)
    return all(_installed(module) for module in modules)


def _core_modules_ready() -> bool:
    return all(_installed(module) for module in RUNTIME_READY_MODULES)


def _install_core_dependencies(
    steps: list[dict[str, object]], warnings: list[str], *, force: bool
) -> None:
    if _env_flag("SFMAPI_INSTANTSFM_SKIP_DEPS"):
        warnings.append(
            "InstantSfM dependency installation was skipped because "
            "SFMAPI_INSTANTSFM_SKIP_DEPS is set."
        )
        steps.append(
            _step("instantsfm_core_dependencies", "skip Python runtime dependencies", "skipped")
        )
        return
    if _core_modules_ready() and not force:
        steps.append(
            _step(
                "instantsfm_core_dependencies",
                "use installed InstantSfM runtime dependencies",
                "skipped",
            )
        )
        return
    packages = _runtime_dependency_packages()
    _run_checked(
        ["uv", "pip", "install", *packages],
        steps,
        name="instantsfm_core_dependencies",
        action="uv pip install " + " ".join(packages),
    )


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) if part.isdigit() else 0 for part in value.split("."))


def _torch_cuda_version() -> str | None:
    try:
        import torch
    except ModuleNotFoundError:
        return None
    return str(torch.version.cuda or "") or None


def _cuda_version_from_path(path: Path) -> str | None:
    name = path.name.removeprefix("v")
    if not name or not name[0].isdigit():
        return None
    return name


def _cuda_candidates() -> list[Path]:
    candidates: list[Path] = []
    for name in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(name)
        if value and Path(value).exists():
            candidates.append(Path(value))
    if os.name != "nt":
        return candidates
    base = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
    cuda_root = base / "NVIDIA GPU Computing Toolkit" / "CUDA"
    if not cuda_root.is_dir():
        return candidates
    candidates.extend(
        path for path in cuda_root.iterdir() if path.is_dir() and path.name.startswith("v")
    )
    return sorted(
        dict.fromkeys(candidates),
        key=lambda path: _version_tuple(_cuda_version_from_path(path) or "0"),
        reverse=True,
    )


def _discover_cuda_home() -> Path | None:
    candidates = _cuda_candidates()
    requested = os.environ.get("SFMAPI_INSTANTSFM_CUDA_VERSION") or _torch_cuda_version()
    if requested:
        for candidate in candidates:
            if _cuda_version_from_path(candidate) == requested:
                return candidate
    return candidates[0] if candidates else None


def _torch_arch_list() -> str | None:
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return os.environ["TORCH_CUDA_ARCH_LIST"]
    try:
        import torch
    except ModuleNotFoundError:
        return None
    if not torch.cuda.is_available():
        return os.environ.get(
            "SFMAPI_INSTANTSFM_DEFAULT_CUDA_ARCH_LIST",
            DEFAULT_TORCH_CUDA_ARCH_LIST,
        )
    major, minor = torch.cuda.get_device_capability(0)
    return f"{major}.{minor}"


def _cuda_build_env(steps: list[dict[str, object]]) -> dict[str, str]:
    env = os.environ.copy()
    if _torch_device() == "cpu":
        steps.append(_step("cuda_build_env", "skip CUDA build env for CPU Torch", "skipped"))
        return env

    cuda_home = _discover_cuda_home()
    if cuda_home is None:
        raise RuntimeError(
            "CUDA_HOME/CUDA_PATH is required to build InstantSfM's native BAE dependency. "
            "Install the CUDA toolkit or set CUDA_HOME before provisioning."
        )
    requested_cuda = os.environ.get("SFMAPI_INSTANTSFM_CUDA_VERSION") or _torch_cuda_version()
    discovered_cuda = _cuda_version_from_path(cuda_home)
    if requested_cuda and discovered_cuda and requested_cuda != discovered_cuda:
        raise RuntimeError(
            "CUDA toolkit version must match the installed CUDA Torch build: "
            f"torch={requested_cuda}, CUDA_HOME={cuda_home} ({discovered_cuda})"
        )
    env["CUDA_HOME"] = str(cuda_home)
    env["CUDA_PATH"] = str(cuda_home)
    os.environ.setdefault("CUDA_HOME", str(cuda_home))
    os.environ.setdefault("CUDA_PATH", str(cuda_home))
    if os.name == "nt":
        env.setdefault("DISTUTILS_USE_SDK", "1")
        os.environ.setdefault("DISTUTILS_USE_SDK", "1")
    arch_list = _torch_arch_list()
    if arch_list:
        env.setdefault("TORCH_CUDA_ARCH_LIST", arch_list)
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch_list)
    steps.append(
        _step(
            "cuda_build_env",
            "configure CUDA_HOME/CUDA_PATH for source builds",
            "done",
            CUDA_HOME=str(cuda_home),
            TORCH_CUDA_ARCH_LIST=env.get("TORCH_CUDA_ARCH_LIST", ""),
            DISTUTILS_USE_SDK=env.get("DISTUTILS_USE_SDK", ""),
        )
    )
    return env


def _install_bae_dependency(
    steps: list[dict[str, object]], warnings: list[str], *, force: bool
) -> None:
    if _env_flag("SFMAPI_INSTANTSFM_SKIP_BAE"):
        warnings.append(
            "BAE installation was skipped; InstantSfM global SfM will not run without bae."
        )
        steps.append(_step("instantsfm_bae_dependency", "skip BAE dependency", "skipped"))
        return
    if _installed("bae") and _installed("pypose") and not force:
        steps.append(
            _step("instantsfm_bae_dependency", "use installed bae/pypose runtime", "skipped")
        )
        return
    env = _cuda_build_env(steps)
    packages = shlex.split(os.environ.get("SFMAPI_INSTANTSFM_BAE_PACKAGES", "")) or list(
        BAE_PACKAGES
    )
    _run_checked(
        ["uv", "pip", "install", "--no-build-isolation", *packages],
        steps,
        name="instantsfm_bae_dependency",
        action="uv pip install --no-build-isolation " + " ".join(packages),
        env=env,
    )


def _install_gaussian_splatting_dependencies(
    steps: list[dict[str, object]], warnings: list[str], *, force: bool
) -> None:
    if not _install_gs_enabled():
        warnings.append(
            "Gaussian splatting dependencies are opt-in because gsplat/fused-ssim "
            "often require host-specific CUDA extension builds. Set "
            "SFMAPI_INSTANTSFM_INSTALL_GS=1 to build them."
        )
        steps.append(
            _step(
                "instantsfm_gs_dependencies",
                "skip optional Gaussian splatting dependencies",
                "skipped",
            )
        )
        return
    if all(_installed(module) for module in GAUSSIAN_SPLATTING_READY_MODULES) and not force:
        steps.append(
            _step(
                "instantsfm_gs_dependencies",
                "use installed Gaussian splatting dependencies",
                "skipped",
            )
        )
        return
    env = _cuda_build_env(steps)
    packages = shlex.split(os.environ.get("SFMAPI_INSTANTSFM_GS_PACKAGES", "")) or list(
        GAUSSIAN_SPLATTING_PACKAGES
    )
    _run_checked(
        ["uv", "pip", "install", "--no-build-isolation", *packages],
        steps,
        name="instantsfm_gs_dependencies",
        action="uv pip install --no-build-isolation " + " ".join(packages),
        env=env,
    )


def _install_runtime(
    root: Path, steps: list[dict[str, object]], warnings: list[str], *, force: bool
) -> bool:
    _patch_instantsfm_source(root, steps)
    _install_instantsfm_source(root, steps, force=force)
    _install_core_dependencies(steps, warnings, force=force)
    _install_bae_dependency(steps, warnings, force=force)
    _install_gaussian_splatting_dependencies(steps, warnings, force=force)
    return _installed_from_root("instantsfm", root) and _runtime_modules_ready()


def provision(*, dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    steps: list[dict[str, object]] = []
    warnings: list[str] = []
    if dry_run:
        return {
            "available": True,
            "provisioned": False,
            "steps": [
                _step(
                    "plugin_source",
                    "reuse local checkout or clone the SceneMap plugin repo with submodules",
                    "planned",
                ),
                _step(
                    "instantsfm_submodule",
                    f"git submodule update --init --recursive {INSTANTSFM_SUBMODULE.as_posix()}",
                    "planned",
                ),
                _step(
                    "instantsfm_source_patches",
                    "patch upstream source compatibility gaps",
                    "planned",
                ),
                _step(
                    "torch_runtime",
                    "uv pip install --reinstall --index-url <torch wheel index> torch torchvision torchaudio",
                    "planned",
                ),
                _step(
                    "instantsfm_source_install",
                    "uv pip install --no-deps -e <third_party/instantsfm>",
                    "planned",
                ),
                _step(
                    "instantsfm_core_dependencies",
                    "uv pip install curated InstantSfM runtime dependencies",
                    "planned",
                ),
                _step(
                    "instantsfm_bae_dependency",
                    "uv pip install --no-build-isolation bae-kai",
                    "planned",
                ),
                _step(
                    "instantsfm_gs_dependencies",
                    "optional: uv pip install --no-build-isolation gsplat fused-ssim",
                    "planned",
                ),
            ],
            "env": {},
            "warnings": [],
        }

    root = _source_root(force=force, steps=steps)
    _ensure_torch_runtime(steps, force=force)
    runtime_ready = _install_runtime(root, steps, warnings, force=force)
    _ensure_torch_runtime(steps, force=False, name="torch_runtime_verify")
    os.environ["SFMAPI_INSTANTSFM_ROOT"] = str(root)
    os.environ.setdefault("SFMAPI_INSTANTSFM_PYTHON", sys.executable)
    return {
        "available": True,
        "provisioned": runtime_ready,
        "steps": steps,
        "env": {
            "SFMAPI_INSTANTSFM_ROOT": str(root),
            "SFMAPI_INSTANTSFM_PYTHON": sys.executable,
            "TORCH_DEVICE": _torch_device(),
        },
        "warnings": warnings,
        "metadata": {
            "source": "submodule",
            "plugin_repo": PLUGIN_REPO,
            "submodule_path": INSTANTSFM_SUBMODULE.as_posix(),
            "dependency_overrides": {
                "scikit-sparse": (
                    "plugin-private SciPy-backed sksparse.cholmod shim "
                    "(PYTHONPATH-injected into worker subprocesses)"
                ),
                "pypose@bae": "bae-kai",
            },
            "optional_groups": {
                "gaussian_splatting": "set SFMAPI_INSTANTSFM_INSTALL_GS=1 to build"
            },
        },
    }
