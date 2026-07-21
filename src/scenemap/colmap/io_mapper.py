"""``ColmapMapper`` — the neutral :class:`sceneio.mapping.Mapper`
conformer for the COLMAP family, built OVER the existing ``run_mapping``
engine internals (it does not fork them).

A COLMAP mapper is the *classical* family: it needs correspondences
(``requires_correspondences=True``) and turns a
:class:`~sceneio.data.CorrespondenceGraph` + views into a sparse
model. This module owns the two translations the neutral contract needs:

1. **graph -> COLMAP database** (:func:`graph_to_colmap_database`): the
   per-image :class:`~sceneio.data.FeatureSet` s become ``keypoints``
   rows, the per-pair :class:`~sceneio.data.PairCorrespondences`
   become ``matches`` + ``two_view_geometries`` rows, and one camera +
   image row is written per view. The ``pair_id`` encoding and schema
   version come from the ``sceneio.colmap_db`` contract (the single
   source of truth for the scene-database shape), so the database this
   writes is the same standard the engine reads.

2. **sparse model -> MappingResult**
   (:func:`reconstruction_to_mapping_result`): COLMAP's *registered*
   image set is mapped back to *input-view positions by name*, so
   ``poses[i] is None`` marks a view COLMAP could not register (the
   first real exercise of the ``SE3 | None`` slot). Cameras become
   per-view calibrations, and the triangulated points become a
   :class:`~sceneio.data.TrackedPointCloud` with real track
   observations.

:class:`ColmapMapper` is a **mixin**: the three provider
``ColmapCliBackend`` classes inherit it so the backend object itself
satisfies the ``Mapper`` protocol (which is what core's ``io_mapper()``
resolver structurally checks). ``map()`` runs ``self.run_mapping(...)``,
i.e. *whichever engine path that provider uses* (subprocess CLI, the C++
in-process extension, or in-process pycolmap).
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sceneio.colmap_db import DATABASE_VERSION_NUMBER, image_pair_to_pair_id
from sceneio.data import (
    SE3,
    Calibration,
    CameraIntrinsics,
    CameraModel,
    CorrespondenceGraph,
    FeatureSet,
    FrameMeta,
    TrackedPointCloud,
    TrackObservation,
    ViewInput,
)
from sceneio.errors import ContractViolation
from sceneio.imagesource import MaterializedImage
from sceneio.mapping import MapperTraits, MappingOptions, MappingResult

# COLMAP TwoViewGeometry::ConfigurationType values we emit.
_CONFIG_CALIBRATED = 2
_CONFIG_UNCALIBRATED = 3

# The COLMAP scene-database tables the translator writes. The columns
# match the sceneio.colmap_db contract; the engine reads keypoints +
# two_view_geometries (the verified scene graph) to build the model.
_CREATE_TABLES: tuple[str, ...] = (
    "CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY NOT NULL, model INTEGER NOT NULL, "
    "width INTEGER NOT NULL, height INTEGER NOT NULL, params BLOB, "
    "prior_focal_length INTEGER NOT NULL)",
    "CREATE TABLE images (image_id INTEGER PRIMARY KEY NOT NULL, name TEXT NOT NULL UNIQUE, "
    "camera_id INTEGER NOT NULL)",
    "CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, "
    "cols INTEGER NOT NULL, data BLOB)",
    "CREATE TABLE matches (pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, "
    "cols INTEGER NOT NULL, data BLOB)",
    "CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, "
    "cols INTEGER NOT NULL, data BLOB, config INTEGER NOT NULL, F BLOB, E BLOB, H BLOB, "
    "qvec BLOB, tvec BLOB)",
)


def view_name(view: ViewInput, index: int) -> str:
    """The stable per-view id shared by the graph keys and the DB rows.

    Matches the id the neutral contract / conformance kit uses to key a
    view's :class:`FeatureSet` (``view.ref`` when set, else the positional
    fallback), so a registered COLMAP image name maps back to the right
    input-view position.
    """
    return view.ref if view.ref is not None else f"view{index:03d}"


def _view_dims(view: ViewInput, features: FeatureSet | None) -> tuple[int, int]:
    """(width, height) for the view's COLMAP camera row."""
    if view.calibration is not None:
        height, width = view.calibration.image_size
        return int(width), int(height)
    image = view.image
    if isinstance(image, np.ndarray):
        return int(image.shape[1]), int(image.shape[0])
    # No calibration and no in-memory pixels (a persisted-image reference):
    # fall back to the keypoint extent so the camera is at least plausibly
    # sized. The real registration path supplies calibration or a database
    # written by extraction; this keeps the translator standalone.
    if features is not None and len(features):
        max_xy = np.max(np.asarray(features.keypoints, dtype=np.float64), axis=0)
        return int(max_xy[0]) + 1, int(max_xy[1]) + 1
    return 1, 1


def _camera_row(view: ViewInput, features: FeatureSet | None) -> tuple[int, int, int, np.ndarray]:
    """(model_id, width, height, params) for one view's camera row."""
    if view.calibration is not None and view.calibration.intrinsics is not None:
        intr = view.calibration.intrinsics
        return (
            int(intr.model.model_id),
            int(intr.width),
            int(intr.height),
            np.asarray(intr.params, dtype="<f8"),
        )
    width, height = _view_dims(view, features)
    focal = 1.2 * max(width, height)
    params = np.array([focal, width / 2.0, height / 2.0], dtype="<f8")
    return int(CameraModel.SIMPLE_PINHOLE.model_id), width, height, params


def _config_for(pair: Any) -> int:
    geometry = pair.geometry
    if geometry is not None:
        if geometry.E is not None:
            return _CONFIG_CALIBRATED
        if geometry.F is not None or geometry.H is not None:
            return _CONFIG_UNCALIBRATED
    return _CONFIG_CALIBRATED


def _matrix_blob(matrix: np.ndarray | None) -> bytes | None:
    if matrix is None:
        return None
    return np.asarray(matrix, dtype="<f8").tobytes()


def graph_to_colmap_database(
    db_path: Path,
    views: Sequence[ViewInput],
    correspondences: CorrespondenceGraph,
    names: Sequence[str] | None = None,
) -> dict[str, int]:
    """Materialize ``views`` + ``correspondences`` into a COLMAP scene database.

    Writes one ``cameras``/``images`` row per view, a ``keypoints`` row
    per view that carries a :class:`FeatureSet`, and ``matches`` +
    ``two_view_geometries`` rows for every *indexed* pair. Detector-free
    (``mode="coordinates"``) pairs have no persistent keypoint indices to
    reference and are rejected — COLMAP-native storage is index-based
    (the coordinate/dense route is a different, non-COLMAP format).

    Returns the ``{image_name: image_id}`` map (1-based ids, in view order).
    """
    view_list = list(views)
    if names is None:
        names = [view_name(view, index) for index, view in enumerate(view_list)]
    else:
        names = list(names)
    name_to_id = {name: index + 1 for index, name in enumerate(names)}

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(f"PRAGMA user_version = {int(DATABASE_VERSION_NUMBER)}")
        for statement in _CREATE_TABLES:
            conn.execute(statement)

        for index, view in enumerate(view_list):
            name = names[index]
            image_id = name_to_id[name]
            features = correspondences.features.get(name)
            model_id, width, height, params = _camera_row(view, features)
            conn.execute(
                "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
                (image_id, model_id, width, height, params.tobytes(), 0),
            )
            conn.execute("INSERT INTO images VALUES (?, ?, ?)", (image_id, name, image_id))
            if features is not None:
                keypoints = np.asarray(features.keypoints, dtype="<f4").reshape(-1, 2)
                conn.execute(
                    "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
                    (image_id, int(keypoints.shape[0]), 2, keypoints.tobytes()),
                )

        for (name_a, name_b), pair in correspondences.pairs.items():
            if pair.mode != "indexed" or pair.indices is None:
                raise ContractViolation(
                    "ColmapMapper: COLMAP-native storage needs indexed "
                    f"correspondences; pair {(name_a, name_b)!r} is mode={pair.mode!r}"
                )
            id_a, id_b = name_to_id[name_a], name_to_id[name_b]
            indices = np.asarray(pair.indices, dtype=np.int64)
            # COLMAP keys matches by the ordered (smaller, larger) image id;
            # the match columns follow that order, so swap when needed.
            ordered = indices if id_a <= id_b else indices[:, ::-1]
            pair_id = image_pair_to_pair_id(id_a, id_b)
            blob = ordered.astype("<u4").tobytes()
            rows = int(ordered.shape[0])
            conn.execute("INSERT INTO matches VALUES (?, ?, ?, ?)", (pair_id, rows, 2, blob))
            geometry = pair.geometry
            conn.execute(
                "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    pair_id,
                    rows,
                    2,
                    blob,
                    _config_for(pair),
                    _matrix_blob(None if geometry is None else geometry.F),
                    _matrix_blob(None if geometry is None else geometry.E),
                    _matrix_blob(None if geometry is None else geometry.H),
                    None,
                    None,
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return name_to_id


def _num_reg_images(rec: Any) -> int:
    value = getattr(rec, "num_reg_images", None)
    if callable(value):
        return int(value())
    if value is not None:
        return int(value)
    return len(getattr(rec, "images", {}) or {})


def _se3_from_image(image: Any) -> SE3:
    """COLMAP world-to-camera image pose -> a cam2world :class:`SE3`.

    ``model.Rotation.quat`` is pycolmap/Eigen order ``(x, y, z, w)``; the
    contract's ``from_colmap_world2cam`` takes ``(w, x, y, z)``.
    """
    qx, qy, qz, qw = image.cam_from_world.rotation.quat
    return SE3.from_colmap_world2cam((qw, qx, qy, qz), image.cam_from_world.translation)


def _calibration_from_camera(camera: Any) -> Calibration | None:
    if camera is None:
        return None
    try:
        model = CameraModel(camera.model_name)
    except (ValueError, AttributeError):
        return None
    intrinsics = CameraIntrinsics(
        model=model,
        width=int(camera.width),
        height=int(camera.height),
        params=np.asarray(list(camera.params), dtype=np.float64),
    )
    return Calibration.from_intrinsics(intrinsics)


def reconstruction_to_mapping_result(
    rec: Any,
    views: Sequence[ViewInput],
    names: Sequence[str],
    *,
    frame: FrameMeta,
    stats: dict[str, Any] | None = None,
) -> MappingResult:
    """Convert an engine sparse model into a view-aligned MappingResult.

    ``rec`` is the duck-typed COLMAP reconstruction produced by the
    engine (``read_colmap_text_model`` output): ``cameras`` / ``images``
    / ``points3D`` dicts. Registered images are matched back to input
    views *by name*; every unregistered view gets ``poses[i] = None`` and
    ``calibrations[i] = None``.
    """
    names = list(names)
    registered = {str(image.name): image for image in rec.images.values()}
    image_id_to_name = {int(image.image_id): str(image.name) for image in rec.images.values()}

    poses: list[SE3 | None] = []
    calibrations: list[Calibration | None] = []
    for index in range(len(names)):
        image = registered.get(names[index])
        if image is None:
            poses.append(None)
            calibrations.append(None)
            continue
        poses.append(_se3_from_image(image))
        calibrations.append(_calibration_from_camera(rec.cameras.get(image.camera_id)))

    geometry = _tracked_point_cloud(rec, image_id_to_name)
    have_calibration = any(calibration is not None for calibration in calibrations)
    return MappingResult(
        poses=tuple(poses),
        frame=frame,
        calibrations=tuple(calibrations) if have_calibration else None,
        geometry=geometry,
        stats=dict(stats or {}),
    )


def _tracked_point_cloud(rec: Any, image_id_to_name: dict[int, str]) -> TrackedPointCloud | None:
    points3d = list(rec.points3D.values())
    if not points3d:
        return None
    xyz = np.array([[float(v) for v in point.xyz] for point in points3d], dtype=np.float64)
    rgb = np.array(
        [[int(c) & 0xFF for c in getattr(point, "color", (0, 0, 0))] for point in points3d],
        dtype=np.uint8,
    )
    tracks: list[tuple[TrackObservation, ...]] = []
    for point in points3d:
        observations: list[TrackObservation] = []
        track = getattr(point, "track", None)
        for element in getattr(track, "elements", []) or []:
            image_id, keypoint_idx = int(element[0]), int(element[1])
            name = image_id_to_name.get(image_id)
            if name is None or keypoint_idx < 0:
                continue
            observations.append(TrackObservation(image_id=name, keypoint_idx=keypoint_idx))
        tracks.append(tuple(observations))
    return TrackedPointCloud(xyz=xyz, rgb=rgb, tracks=tuple(tracks))


class ColmapMapper:
    """Mixin making a COLMAP backend a neutral ``sceneio`` ``Mapper``.

    The three provider ``ColmapCliBackend`` classes inherit this so the
    backend instance itself satisfies the ``Mapper`` protocol that core's
    ``io_mapper()`` resolver structurally checks. ``map()`` reuses the
    concrete backend's own engine path via ``self.run_mapping(...)``.
    """

    def traits(self) -> MapperTraits:
        return MapperTraits(
            requires_correspondences=True,
            accepts_pose_priors=True,
            accepts_depth_priors=False,
            # COLMAP consumes prior intrinsics through its camera database
            # rows (written by the graph translator from ViewInput.calibration).
            accepts_calibration=True,
            emits_dense=False,
            # COLMAP is only metric when anchored to a metric prior/GPS; the
            # capability is honest (see _frame_meta for the per-run claim).
            metric_capable=True,
        )

    def map(
        self,
        views: Sequence[ViewInput],
        *,
        correspondences: CorrespondenceGraph | None = None,
        options: MappingOptions | None = None,
    ) -> MappingResult:
        traits = self.traits()
        if traits.requires_correspondences and correspondences is None:
            raise ContractViolation(
                "ColmapMapper requires a correspondence graph "
                "(traits.requires_correspondences=True); got correspondences=None"
            )
        view_list = list(views)
        if not view_list:
            raise ContractViolation("ColmapMapper.map: at least one view is required")
        assert correspondences is not None  # narrowed by the guard above
        options = options or MappingOptions()
        names = [view_name(view, index) for index, view in enumerate(view_list)]
        kind = str(options.extra.get("kind", "incremental"))

        workdir = Path(tempfile.mkdtemp(prefix="colmap_mapper_"))
        try:
            db_path = workdir / "database.db"
            sparse_root = workdir / "sparse"
            job_dir = workdir / "job"
            sparse_root.mkdir(parents=True, exist_ok=True)
            job_dir.mkdir(parents=True, exist_ok=True)

            graph_to_colmap_database(db_path, view_list, correspondences, names)
            image_root = self._resolve_image_root(view_list, names, workdir)
            pose_priors = (
                self._pose_priors_arg(view_list, names) if traits.accepts_pose_priors else None
            )
            spec = dict(options.extra)
            spec.pop("kind", None)
            if options.seed is not None:
                spec.setdefault("random_seed", int(options.seed))
            if options.max_views is not None:
                spec.setdefault("max_num_images", int(options.max_views))

            summaries, reconstructions = self.run_mapping(  # type: ignore[attr-defined]
                kind=kind,
                db_path=db_path,
                image_root=image_root,
                sparse_root=sparse_root,
                job_dir=job_dir,
                spec=spec,
                pose_priors=pose_priors,
            )
            if not reconstructions:
                raise ContractViolation(
                    "ColmapMapper.map: the engine registered no reconstruction "
                    "(no view could be registered from the given correspondences)"
                )
            rec = max(reconstructions, key=_num_reg_images)
            frame = self._frame_meta(view_list, kind, traits)
            stats = {
                "engine": "colmap.run_mapping",
                "kind": kind,
                "num_submodels": len(reconstructions),
                "num_input_views": len(view_list),
            }
            if summaries:
                stats["summaries"] = summaries
            return reconstruction_to_mapping_result(rec, view_list, names, frame=frame, stats=stats)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    # -- helpers ---------------------------------------------------------

    def _frame_meta(self, views: Sequence[ViewInput], kind: str, traits: MapperTraits) -> FrameMeta:
        metric = (
            traits.metric_capable
            and traits.accepts_pose_priors
            and kind == "incremental"
            and any(v.pose_prior is not None and v.pose_prior.is_metric for v in views)
        )
        if metric:
            # world_frame stays COLMAP's gauge; a metric prior fixes only scale.
            return FrameMeta(
                world_frame="colmap", scale="metric", scale_provenance="prior_anchored"
            )
        return FrameMeta(world_frame="colmap", scale="arbitrary", scale_provenance="unknown")

    def _pose_priors_arg(
        self, views: Sequence[ViewInput], names: Sequence[str]
    ) -> dict[str, Any] | None:
        priors: dict[str, Any] = {}
        for name, view in zip(names, views, strict=True):
            if view.pose_prior is None:
                continue
            # COLMAP stores the camera *position* in world; for a cam2world
            # pose that is the translation.
            pose = view.pose_prior.pose.as_convention("opencv_cam2world")
            priors[name] = {
                "position": [float(v) for v in pose.translation],
                "is_metric": bool(view.pose_prior.is_metric),
            }
        return {"priors": priors} if priors else None

    def _resolve_image_root(
        self, views: Sequence[ViewInput], names: Sequence[str], workdir: Path
    ) -> Path:
        materialized = [view.image for view in views if isinstance(view.image, MaterializedImage)]
        if materialized and len(materialized) == len(views):
            parents = [Path(image.abs_path).parent for image in materialized]
            try:
                return Path(os.path.commonpath([str(parent) for parent in parents]))
            except ValueError:
                return parents[0]
        # In-memory views: COLMAP sparse mapping reads geometry from the
        # database (cameras + keypoints + two_view_geometries), not pixels, so
        # a directory of placeholder files by name is enough for the engine
        # call to resolve an image root.
        image_dir = workdir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for name, view in zip(names, views, strict=True):
            self._write_placeholder_image(image_dir / name, view.image)
        return image_dir

    @staticmethod
    def _write_placeholder_image(path: Path, image: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(image, np.ndarray):
            try:
                from PIL import Image as PILImage
            except ImportError:
                path.write_bytes(b"")
                return
            PILImage.fromarray(image).save(path.with_suffix(path.suffix or ".png"))
            return
        path.write_bytes(b"")


__all__ = [
    "ColmapMapper",
    "graph_to_colmap_database",
    "reconstruction_to_mapping_result",
    "view_name",
]
