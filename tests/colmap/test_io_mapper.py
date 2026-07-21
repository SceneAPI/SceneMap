"""Tests for the COLMAP ``ColmapMapper`` sceneio conformer.

The engine is replaced with a canned in-memory reconstruction built FROM
the database the graph translator wrote, so these run without pycolmap or
a COLMAP executable (they land in the default CI lane). One
``needs_pycolmap`` test proves the real provider backend presents as a
conforming ``Mapper``.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest
from sceneio.colmap_db import image_pair_to_pair_id
from sceneio.data import (
    SE3,
    CameraModel,
    CorrespondenceGraph,
    PairCorrespondences,
    PosePrior,
    ViewInput,
)
from sceneio.errors import ContractViolation
from sceneio.mapping import Mapper, MapperTraits, MappingOptions, MappingResult
from sceneio.testing import (
    assert_mapper_conformance,
    make_synthetic_correspondence_graph,
    make_synthetic_views,
)

from scenemap.colmap.cli.backend import ColmapCliBackend
from scenemap.colmap.io_mapper import (
    ColmapMapper,
    graph_to_colmap_database,
    view_name,
)
from scenemap.colmap.model import (
    Camera,
    Image,
    Point3D,
    Reconstruction,
    Rigid3,
    Rotation,
    Track,
)

_MODEL_ID_TO_NAME = {model.model_id: model.value for model in CameraModel}


def _reconstruction_from_db(
    db_path, *, skip_names=frozenset(), num_points: int = 3
) -> Reconstruction:
    """Build a canned reconstruction from the translator-written database.

    Registers every image whose name is not in ``skip_names`` (so callers
    can force unregistered views), copying the camera rows back out and
    inventing deterministic poses + tracked points that reference the
    registered images. Reading the database here also proves the graph
    translator wrote readable ``images`` / ``cameras`` rows.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        image_rows = conn.execute(
            "SELECT image_id, name, camera_id FROM images ORDER BY image_id"
        ).fetchall()
        camera_rows = {
            int(camera_id): (int(model), int(width), int(height), params)
            for camera_id, model, width, height, params in conn.execute(
                "SELECT camera_id, model, width, height, params FROM cameras"
            )
        }
    finally:
        conn.close()

    rec = Reconstruction()
    registered_ids: list[int] = []
    for image_id, name, camera_id in image_rows:
        if name in skip_names:
            continue
        registered_ids.append(int(image_id))
        model_id, width, height, params_blob = camera_rows[int(camera_id)]
        rec.cameras[int(camera_id)] = Camera(
            camera_id=int(camera_id),
            model_name=_MODEL_ID_TO_NAME[model_id],
            width=width,
            height=height,
            params=list(np.frombuffer(params_blob, dtype="<f8")),
        )
        rec.images[int(image_id)] = Image(
            image_id=int(image_id),
            name=str(name),
            camera_id=int(camera_id),
            cam_from_world=Rigid3(
                rotation=Rotation(quat=(0.0, 0.0, 0.0, 1.0)),
                translation=(float(image_id), 0.0, 0.0),
            ),
        )
    for point_index in range(num_points):
        rec.points3D[point_index + 1] = Point3D(
            point3D_id=point_index + 1,
            xyz=(float(point_index), 1.0, 2.0),
            color=(255, 0, 0),
            track=Track(elements=[(image_id, point_index) for image_id in registered_ids]),
        )
    return rec


class _FakeEngineColmapBackend(ColmapCliBackend):
    """The real CLI provider backend with only its engine call replaced."""

    def __init__(self, *, skip_names=frozenset(), num_points: int = 3) -> None:
        super().__init__()
        self._skip_names = frozenset(skip_names)
        self._num_points = num_points

    def run_mapping(
        self,
        *,
        kind,
        db_path,
        image_root,
        sparse_root,
        job_dir,
        spec,
        pose_priors=None,
        progress=None,
    ):
        rec = _reconstruction_from_db(
            db_path, skip_names=self._skip_names, num_points=self._num_points
        )
        summaries = [
            {"idx": 0, "num_reg_images": rec.num_reg_images(), "num_points3D": len(rec.points3D)}
        ]
        return summaries, [rec]


def _views_with_metric_priors(count: int = 3) -> list[ViewInput]:
    views: list[ViewInput] = []
    for index in range(count):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        pose = SE3(np.eye(3), np.array([float(index), 0.0, 0.0]))
        views.append(
            ViewInput(
                image=image,
                name=f"view{index:03d}",
                pose_prior=PosePrior(pose=pose, is_metric=True),
            )
        )
    return views


# --- graph -> database translation ----------------------------------------


def test_graph_to_colmap_database_writes_readable_rows(tmp_path):
    views = make_synthetic_views(3)
    graph = make_synthetic_correspondence_graph(views)
    db_path = tmp_path / "database.db"

    name_to_id = graph_to_colmap_database(db_path, views, graph)
    assert name_to_id == {"view000": 1, "view001": 2, "view002": 3}

    conn = sqlite3.connect(str(db_path))
    try:
        assert conn.execute("SELECT count(*) FROM cameras").fetchone()[0] == 3
        assert conn.execute("SELECT count(*) FROM images").fetchone()[0] == 3
        # keypoints round-trip as float32 (x, y) rows.
        rows, cols, data = conn.execute(
            "SELECT rows, cols, data FROM keypoints WHERE image_id = 1"
        ).fetchone()
        assert (rows, cols) == (6, 2)
        decoded = np.frombuffer(data, dtype="<f4").reshape(rows, cols)
        np.testing.assert_allclose(
            decoded, np.asarray(graph.features["view000"].keypoints), rtol=1e-5
        )
        # the chain graph has two pairs; both get matches + two_view_geometries.
        assert conn.execute("SELECT count(*) FROM matches").fetchone()[0] == 2
        assert conn.execute("SELECT count(*) FROM two_view_geometries").fetchone()[0] == 2
        # pair ids use the contract's encoding.
        pair_id = image_pair_to_pair_id(1, 2)
        match = conn.execute(
            "SELECT rows, cols, data FROM matches WHERE pair_id = ?", (pair_id,)
        ).fetchone()
        assert match is not None
        indices = np.frombuffer(match[2], dtype="<u4").reshape(match[0], match[1])
        assert indices.shape == (6, 2)
        config = conn.execute(
            "SELECT config FROM two_view_geometries WHERE pair_id = ?", (pair_id,)
        ).fetchone()[0]
        assert config in (2, 3)
    finally:
        conn.close()


def test_graph_to_colmap_database_rejects_detector_free(tmp_path):
    views = make_synthetic_views(2)
    coords = np.zeros((3, 2), dtype=np.float32)
    graph = CorrespondenceGraph(
        features={},
        pairs={("view000", "view001"): PairCorrespondences.from_coordinates(coords, coords)},
    )
    with pytest.raises(ContractViolation, match="indexed"):
        graph_to_colmap_database(tmp_path / "database.db", views, graph)


# --- ColmapMapper.map() ----------------------------------------------------


def test_map_aligns_poses_to_views_with_unregistered_nones():
    views = make_synthetic_views(4)
    graph = make_synthetic_correspondence_graph(views)
    backend = _FakeEngineColmapBackend(skip_names={"view001", "view003"})

    result = backend.map(views, correspondences=graph)

    assert isinstance(result, MappingResult)
    assert len(result.poses) == 4
    assert result.poses[0] is not None
    assert result.poses[1] is None
    assert result.poses[2] is not None
    assert result.poses[3] is None
    assert list(result.registered_mask) == [True, False, True, False]
    # calibrations stay index-aligned, None where the view is unregistered.
    assert result.calibrations is not None
    assert result.calibrations[0] is not None
    assert result.calibrations[1] is None
    assert result.calibrations[3] is None
    assert result.calibrations[0].intrinsics.model is CameraModel.SIMPLE_PINHOLE
    # the sparse cloud carries real track observations keyed by view name.
    assert result.geometry is not None
    assert len(result.geometry) == 3
    observed = {obs.image_id for track in result.geometry.tracks for obs in track}
    assert observed == {"view000", "view002"}


def test_map_requires_correspondences():
    views = make_synthetic_views(3)
    backend = _FakeEngineColmapBackend()
    with pytest.raises(ContractViolation, match="requires a correspondence graph"):
        backend.map(views, correspondences=None)


def test_map_claims_metric_only_when_prior_anchored():
    backend = _FakeEngineColmapBackend()

    metric_views = _views_with_metric_priors(3)
    metric_result = backend.map(
        metric_views,
        correspondences=make_synthetic_correspondence_graph(metric_views),
        options=MappingOptions(extra={"kind": "incremental"}),
    )
    assert metric_result.frame.scale == "metric"
    assert metric_result.frame.scale_provenance == "prior_anchored"

    plain_views = make_synthetic_views(3)
    plain_result = backend.map(
        plain_views, correspondences=make_synthetic_correspondence_graph(plain_views)
    )
    assert plain_result.frame.scale == "arbitrary"
    assert plain_result.frame.scale_provenance == "unknown"


def test_view_name_matches_graph_keys():
    views = make_synthetic_views(2)
    assert [view_name(view, index) for index, view in enumerate(views)] == ["view000", "view001"]


# --- conformance kit -------------------------------------------------------


def test_conformance_fake_engine():
    """The fake-engine variant always runs — full sceneio conformance."""
    backend = _FakeEngineColmapBackend(skip_names={"view002"})
    result = assert_mapper_conformance(backend)
    # view002 was deliberately left unregistered; the kit tolerates the None.
    assert result.poses[2] is None
    assert result.frame.scale == "arbitrary"


@pytest.mark.needs_pycolmap
def test_real_pycolmap_backend_is_a_conforming_mapper():
    """Engine-gated: the real provider presents as a conforming ``Mapper``.

    A full ``assert_mapper_conformance`` registration pass is not possible
    against the kit's synthetic random-keypoint graph (real COLMAP cannot
    triangulate it), so this proves the protocol/traits conformance and the
    ``requires_correspondences`` guard, which run before the engine.
    """
    pytest.importorskip("pycolmap")
    from scenemap.colmap.pycolmap_backend import PycolmapBackend

    backend = PycolmapBackend()
    assert isinstance(backend, Mapper)
    traits = backend.traits()
    assert isinstance(traits, MapperTraits)
    assert traits.requires_correspondences is True
    assert traits.emits_dense is False
    with pytest.raises(ContractViolation):
        backend.map(make_synthetic_views(2), correspondences=None)


def test_colmap_mapper_is_a_mixin_on_all_providers():
    from scenemap.colmap.native.backend import ColmapCliBackend as NativeBackend
    from scenemap.colmap.pycolmap.backend import ColmapCliBackend as PycolmapCliBackend

    for backend_cls in (ColmapCliBackend, NativeBackend, PycolmapCliBackend):
        assert issubclass(backend_cls, ColmapMapper)
        assert isinstance(backend_cls().traits(), MapperTraits)
