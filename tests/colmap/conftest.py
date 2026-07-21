from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import NamedTuple

import pytest

from scenemap.colmap.pycolmap.backend import ColmapCliBackend, colmap_runtime_env
from scenemap.colmap.pycolmap_backend import register_pycolmap_dll_directories

SOUTH_BUILDING_URL = "https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def pytest_configure(config: pytest.Config) -> None:
    register_pycolmap_dll_directories()


class SampleSubset(NamedTuple):
    image_root: Path
    image_names: list[str]


def _image_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _resolve_sample_image_root(path: Path) -> Path:
    candidates = [
        path,
        path / "images",
        path / "south-building" / "images",
        path / "South-Building" / "images",
    ]
    for candidate in candidates:
        if candidate.is_dir() and _image_files(candidate):
            return candidate
    raise FileNotFoundError(f"no COLMAP sample images found under {path}")


def _download_south_building(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "south-building.zip"
    extracted = cache_dir / "south-building"
    if not _image_files(extracted):
        if not archive.exists():
            urllib.request.urlretrieve(SOUTH_BUILDING_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extracted)
    return _resolve_sample_image_root(extracted)


@pytest.fixture(scope="session")
def colmap_executable() -> Path:
    exe = ColmapCliBackend()._find_colmap()
    if exe is None:
        pytest.skip(
            "COLMAP executable not found. Set SFMAPI_COLMAP_EXECUTABLE or put colmap on PATH."
        )
    try:
        result = subprocess.run(
            [str(exe), "-h"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except OSError as exc:
        pytest.skip(f"COLMAP executable is not runnable on this platform: {exe} ({exc})")
    if result.returncode != 0 or "COLMAP" not in result.stdout + result.stderr:
        pytest.skip(f"COLMAP executable did not report a valid COLMAP help surface: {exe}")
    os.environ["PATH"] = colmap_runtime_env(exe)["PATH"]
    return exe


@pytest.fixture(scope="session")
def colmap_sample_images(pytestconfig: pytest.Config) -> Path:
    explicit = os.environ.get("SFMAPI_COLMAP_SAMPLE_DATA")
    if explicit:
        return _resolve_sample_image_root(Path(explicit))

    if os.environ.get("SFMAPI_COLMAP_DOWNLOAD_SAMPLE") == "1":
        cache_dir = Path(str(pytestconfig.cache.makedir("colmap-sample-data")))
        return _download_south_building(cache_dir)

    pytest.skip(
        "COLMAP sample data not configured. Set SFMAPI_COLMAP_SAMPLE_DATA to an extracted "
        "South Building dataset, or set SFMAPI_COLMAP_DOWNLOAD_SAMPLE=1 to download it."
    )


@pytest.fixture
def colmap_sample_subset(tmp_path: Path, colmap_sample_images: Path) -> SampleSubset:
    limit = int(os.environ.get("SFMAPI_COLMAP_SAMPLE_IMAGE_LIMIT", "8"))
    files = _image_files(colmap_sample_images)
    if len(files) < 2:
        pytest.skip(f"COLMAP sample data needs at least 2 images; found {len(files)}")
    selected = files[: max(2, limit)]
    image_root = tmp_path / "sample-images"
    image_root.mkdir()
    image_names: list[str] = []
    for src in selected:
        dst = image_root / src.name
        shutil.copy2(src, dst)
        image_names.append(dst.name)
    return SampleSubset(image_root=image_root, image_names=image_names)
