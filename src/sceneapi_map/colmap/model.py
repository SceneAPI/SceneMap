"""Minimal COLMAP text-model parser.

The sfmapi snapshot writer consumes a small duck-typed subset of
pycolmap.Reconstruction. This module builds that shape from COLMAP's text
export without importing pycolmap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Rotation:
    # sfmapi's emitter expects pycolmap's Eigen order: x, y, z, w.
    quat: tuple[float, float, float, float]


@dataclass
class Rigid3:
    rotation: Rotation
    translation: tuple[float, float, float]


@dataclass
class Camera:
    camera_id: int
    model_name: str
    width: int
    height: int
    params: list[float]
    has_prior_focal_length: bool = False


@dataclass
class Point2D:
    xy: tuple[float, float]
    point3D_id: int | None


@dataclass
class Image:
    image_id: int
    name: str
    camera_id: int
    cam_from_world: Rigid3
    points2D: list[Point2D] = field(default_factory=list)


@dataclass
class Track:
    elements: list[tuple[int, int]]


@dataclass
class Point3D:
    point3D_id: int
    xyz: tuple[float, float, float]
    color: tuple[int, int, int]
    track: Track


@dataclass
class Reconstruction:
    cameras: dict[int, Camera] = field(default_factory=dict)
    images: dict[int, Image] = field(default_factory=dict)
    points3D: dict[int, Point3D] = field(default_factory=dict)
    rigs: dict[int, object] = field(default_factory=dict)
    frames: dict[int, object] = field(default_factory=dict)

    def num_reg_images(self) -> int:
        return len(self.images)


def _iter_data_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _parse_cameras(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    for line in _iter_data_lines(path):
        parts = line.split()
        if len(parts) < 5:
            continue
        camera_id = int(parts[0])
        cameras[camera_id] = Camera(
            camera_id=camera_id,
            model_name=parts[1],
            width=int(parts[2]),
            height=int(parts[3]),
            params=[float(x) for x in parts[4:]],
        )
    return cameras


def _parse_images(path: Path) -> dict[int, Image]:
    lines = _iter_data_lines(path)
    images: dict[int, Image] = {}
    idx = 0
    while idx < len(lines):
        header = lines[idx].split()
        idx += 1
        if len(header) < 10:
            continue

        image_id = int(header[0])
        qw, qx, qy, qz = (float(v) for v in header[1:5])
        tx, ty, tz = (float(v) for v in header[5:8])
        camera_id = int(header[8])
        name = " ".join(header[9:])

        points2d: list[Point2D] = []
        if idx < len(lines):
            point_parts = lines[idx].split()
            idx += 1
            for offset in range(0, len(point_parts), 3):
                if offset + 2 >= len(point_parts):
                    break
                x = float(point_parts[offset])
                y = float(point_parts[offset + 1])
                point_id_raw = int(point_parts[offset + 2])
                point_id = None if point_id_raw < 0 else point_id_raw
                points2d.append(Point2D(xy=(x, y), point3D_id=point_id))

        images[image_id] = Image(
            image_id=image_id,
            name=name,
            camera_id=camera_id,
            cam_from_world=Rigid3(
                rotation=Rotation(quat=(qx, qy, qz, qw)),
                translation=(tx, ty, tz),
            ),
            points2D=points2d,
        )
    return images


def _parse_points3d(path: Path) -> dict[int, Point3D]:
    points: dict[int, Point3D] = {}
    for line in _iter_data_lines(path):
        parts = line.split()
        if len(parts) < 8:
            continue
        point_id = int(parts[0])
        track_values = parts[8:]
        track: list[tuple[int, int]] = []
        for offset in range(0, len(track_values), 2):
            if offset + 1 >= len(track_values):
                break
            track.append((int(track_values[offset]), int(track_values[offset + 1])))
        points[point_id] = Point3D(
            point3D_id=point_id,
            xyz=(float(parts[1]), float(parts[2]), float(parts[3])),
            color=(int(parts[4]), int(parts[5]), int(parts[6])),
            track=Track(elements=track),
        )
    return points


def read_colmap_text_model(path: Path) -> Reconstruction:
    """Read a COLMAP text model directory."""
    model_dir = Path(path)
    return Reconstruction(
        cameras=_parse_cameras(model_dir / "cameras.txt"),
        images=_parse_images(model_dir / "images.txt"),
        points3D=_parse_points3d(model_dir / "points3D.txt"),
    )


__all__ = [
    "Camera",
    "Image",
    "Point2D",
    "Point3D",
    "Reconstruction",
    "Rigid3",
    "Rotation",
    "Track",
    "read_colmap_text_model",
]
