"""Tests for the MapAnything feed-forward ``Mapper`` conformer.

The single real-model call (``backend._run_inference``) is monkeypatched with
canned numpy predictions, so every test here runs with NO torch / mapanything
/ weights (they land in the default CI lane). The whole adapter AROUND the
model call — input build, output conversion, point-cloud fusion, frame/scale
logic, weights resolution — is exercised for real. One ``needs_mapanything``
test proves the real engine path when it is installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from sceneapi_io.data import (
    SE3,
    Calibration,
    CameraIntrinsics,
    CameraModel,
    ConfidenceMap,
    DepthMap,
    Pointmap,
    PosePrior,
    RayMap,
    ViewInput,
)
from sceneapi_io.mapping import Mapper, MapperTraits, MappingOptions, MappingResult
from sceneapi_io.testing import (
    assert_mapper_conformance,
    make_synthetic_correspondence_graph,
    make_synthetic_views,
)

import sceneapi_map.mapanything.backend as backend_mod
from sceneapi_map.mapanything.backend import (
    APACHE_WEIGHTS,
    CC_BY_NC_WEIGHTS,
    DEFAULT_WEIGHTS,
    WEIGHTS_ENV_VAR,
    MapAnythingBackend,
    build_view_payload,
    resolve_weights,
)

# --- canned inference ------------------------------------------------------


def _canned_pred(
    height: int,
    width: int,
    *,
    pose: np.ndarray | None = None,
    conf_peak: float = 0.8,
    with_color: bool = True,
) -> dict[str, np.ndarray]:
    """A deterministic per-view prediction shaped like MapAnything's output."""
    rng = np.random.default_rng(height * 1000 + width)
    pred: dict[str, np.ndarray] = {
        "pts3d": rng.random((height, width, 3)).astype(np.float32),
        "conf": np.full((height, width), conf_peak, dtype=np.float32),
        "intrinsics": np.array(
            [[float(width), 0.0, width / 2.0], [0.0, float(height), height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "mask": np.ones((height, width), dtype=bool),
    }
    if pose is not None:
        pred["camera_poses"] = pose
    else:
        pred["camera_poses"] = np.eye(4, dtype=np.float64)
    if with_color:
        pred["img_no_norm"] = np.full((height, width, 3), 127.0, dtype=np.float32)
    return pred


def _install_mock(monkeypatch, pred_for=None) -> None:
    """Patch the single lazy inference fn with a canned one-pred-per-view mock."""

    def _mock_infer(payloads, *, weights, options):
        out = []
        for index, payload in enumerate(payloads):
            height, width = payload["img"].shape[:2]
            if pred_for is not None:
                out.append(pred_for(index, height, width))
            else:
                out.append(_canned_pred(height, width))
        return out

    monkeypatch.setattr(backend_mod, "_run_inference", _mock_infer)


# --- (a) sceneapi-io conformance (always runs) -----------------------------


def test_mock_inference_conformance(monkeypatch):
    """Full sceneapi-io mapper conformance against the real backend + mock engine."""
    _install_mock(monkeypatch)
    backend = MapAnythingBackend()

    result = assert_mapper_conformance(backend)

    assert isinstance(result, MappingResult)
    # Feed-forward registers all synthetic views.
    assert all(pose is not None for pose in result.poses)


def test_backend_is_a_mapper(monkeypatch):
    backend = MapAnythingBackend()
    assert isinstance(backend, Mapper)
    assert isinstance(backend.traits(), MapperTraits)


# --- (b) correspondences optional (accepts None AND a graph) ---------------


def test_map_accepts_none_and_a_graph(monkeypatch):
    """requires_correspondences=False: both None and a supplied graph succeed."""
    _install_mock(monkeypatch)
    backend = MapAnythingBackend()
    views = make_synthetic_views(3)

    none_result = backend.map(views, correspondences=None)
    graph = make_synthetic_correspondence_graph(views)
    graph_result = backend.map(views, correspondences=graph)

    assert isinstance(none_result, MappingResult)
    assert isinstance(graph_result, MappingResult)
    # The graph is ignored — same number of registered views either way.
    assert len(none_result.poses) == len(graph_result.poses) == 3


# --- (c) traits honesty ----------------------------------------------------


def test_traits_are_honest_feed_forward():
    traits = MapAnythingBackend().traits()
    assert traits.requires_correspondences is False
    assert traits.accepts_pose_priors is True
    assert traits.accepts_depth_priors is True
    assert traits.accepts_calibration is True
    assert traits.emits_dense is True
    assert traits.metric_capable is True


def test_capabilities_advertise_feed_forward():
    assert MapAnythingBackend().capabilities() == {"map.feed_forward"}


# --- (d) metric-scale propagation ------------------------------------------


def _views_with_metric_priors(count: int = 3) -> list[ViewInput]:
    views: list[ViewInput] = []
    for index in range(count):
        pose = SE3(np.eye(3), np.array([float(index), 0.0, 0.0]))
        views.append(
            ViewInput(
                image=np.zeros((8, 8, 3), dtype=np.uint8),
                name=f"view{index:03d}",
                pose_prior=PosePrior(pose=pose, is_metric=True),
            )
        )
    return views


def test_metric_prior_yields_prior_anchored_metric(monkeypatch):
    _install_mock(monkeypatch)
    backend = MapAnythingBackend()

    result = backend.map(_views_with_metric_priors(3), correspondences=None)

    assert result.frame.scale == "metric"
    assert result.frame.scale_provenance == "prior_anchored"


def test_no_prior_apache_default_is_not_metric(monkeypatch):
    _install_mock(monkeypatch)
    backend = MapAnythingBackend()

    result = backend.map(make_synthetic_views(3), correspondences=None)

    assert result.frame.scale != "metric"
    assert result.frame.scale == "normalized"
    assert result.stats["weights"] == APACHE_WEIGHTS


def test_trust_model_metric_opt_in_yields_model_claimed_metric(monkeypatch):
    _install_mock(monkeypatch)
    backend = MapAnythingBackend()

    result = backend.map(
        make_synthetic_views(2),
        correspondences=None,
        options=MappingOptions(extra={"trust_model_metric": True}),
    )

    assert result.frame.scale == "metric"
    assert result.frame.scale_provenance == "model_claimed"


# --- (e) all views registered in the happy path; None-tolerant otherwise ---


def test_happy_path_registers_all_views(monkeypatch):
    _install_mock(monkeypatch)
    backend = MapAnythingBackend()

    result = backend.map(make_synthetic_views(4), correspondences=None)

    assert list(result.registered_mask) == [True, True, True, True]
    assert result.calibrations is not None
    assert all(calibration is not None for calibration in result.calibrations)


def test_map_is_none_tolerant_for_unregistered_view(monkeypatch):
    """A prediction missing camera_poses -> that view is honestly unregistered."""

    def _pred_for(index, height, width):
        if index == 1:
            pred = _canned_pred(height, width)
            del pred["camera_poses"]
            return pred
        return _canned_pred(height, width)

    _install_mock(monkeypatch, pred_for=_pred_for)
    backend = MapAnythingBackend()

    result = backend.map(make_synthetic_views(3), correspondences=None)

    assert result.poses[0] is not None
    assert result.poses[1] is None
    assert result.poses[2] is not None
    assert result.dense is not None
    assert result.dense[1] is None


# --- (f) dense payload shape -----------------------------------------------


def test_dense_payload_shape(monkeypatch):
    _install_mock(monkeypatch)
    backend = MapAnythingBackend()

    result = backend.map(make_synthetic_views(2), correspondences=None)

    assert result.dense is not None
    assert len(result.dense) == 2
    for entry in result.dense:
        assert entry is not None
        pointmap, confidence = entry
        assert isinstance(pointmap, Pointmap)
        assert isinstance(confidence, ConfidenceMap)
        assert pointmap.shape == (8, 8)
        assert confidence.shape == (8, 8)
        assert pointmap.points.dtype == np.float32
        assert pointmap.frame == "world"


def test_fused_geometry_is_present(monkeypatch):
    _install_mock(monkeypatch)
    backend = MapAnythingBackend()

    result = backend.map(make_synthetic_views(3), correspondences=None)

    assert result.geometry is not None
    # 3 views x 8x8 fully-valid pixels.
    assert len(result.geometry) == 3 * 8 * 8
    assert result.geometry.rgb is not None
    # Feed-forward has no correspondences -> no track observations.
    assert result.geometry.tracks is None


def test_confidence_out_of_range_is_normalized(monkeypatch):
    """MapAnything conf can exceed 1; the ConfidenceMap must stay in [0, 1]."""

    def _pred_for(index, height, width):
        pred = _canned_pred(height, width, conf_peak=5.0)
        pred["conf"][0, 0] = np.nan  # non-finite pixels must be tolerated too
        return pred

    _install_mock(monkeypatch, pred_for=_pred_for)
    backend = MapAnythingBackend()

    result = backend.map(make_synthetic_views(1), correspondences=None)

    assert result.dense is not None
    _, confidence = result.dense[0]
    assert float(confidence.values.max()) <= 1.0
    assert float(confidence.values.min()) >= 0.0
    assert np.isfinite(confidence.values).all()


def test_fused_cloud_subsamples_to_cap(monkeypatch):
    _install_mock(monkeypatch)
    backend = MapAnythingBackend()

    result = backend.map(
        make_synthetic_views(3),
        correspondences=None,
        options=MappingOptions(extra={"max_points": 50}),
    )

    assert result.geometry is not None
    assert len(result.geometry) == 50


# --- input build -----------------------------------------------------------


def test_build_view_payload_attaches_accepted_priors():
    intrinsics = CameraIntrinsics(
        model=CameraModel.PINHOLE,
        width=8,
        height=6,
        params=np.array([100.0, 120.0, 4.0, 3.0]),
    )
    view = ViewInput(
        image=np.zeros((6, 8, 3), dtype=np.uint8),
        name="v",
        calibration=Calibration.from_intrinsics(intrinsics),
        pose_prior=PosePrior(pose=SE3.identity(), is_metric=True),
        depth_prior=DepthMap(depth=np.ones((6, 8), dtype=np.float32)),
    )

    payload = build_view_payload(view, MapAnythingBackend().traits())

    assert payload["img"].shape == (6, 8, 3)
    np.testing.assert_allclose(
        payload["intrinsics"], [[100.0, 0.0, 4.0], [0.0, 120.0, 3.0], [0.0, 0.0, 1.0]]
    )
    assert payload["camera_poses"].shape == (4, 4)
    assert payload["is_metric_scale"] is True
    assert payload["depth_z"].shape == (6, 8)


def test_build_view_payload_accepts_ray_calibration():
    directions = np.zeros((5, 4, 3), dtype=np.float32)
    directions[..., 2] = 1.0  # unit +z rays
    view = ViewInput(
        image=np.zeros((5, 4, 3), dtype=np.uint8),
        calibration=Calibration.from_rays(RayMap(directions=directions)),
    )

    payload = build_view_payload(view, MapAnythingBackend().traits())

    assert "ray_directions" in payload
    assert "intrinsics" not in payload
    assert payload["ray_directions"].shape == (5, 4, 3)


def test_grayscale_image_is_expanded_to_rgb():
    view = ViewInput(image=np.zeros((4, 4), dtype=np.uint8))
    payload = build_view_payload(view, MapAnythingBackend().traits())
    assert payload["img"].shape == (4, 4, 3)


# --- weights / license -----------------------------------------------------


def test_apache_weights_are_the_default(monkeypatch):
    monkeypatch.delenv(WEIGHTS_ENV_VAR, raising=False)
    assert DEFAULT_WEIGHTS == APACHE_WEIGHTS
    assert DEFAULT_WEIGHTS != CC_BY_NC_WEIGHTS
    assert resolve_weights(None) == APACHE_WEIGHTS
    assert resolve_weights(MappingOptions()) == APACHE_WEIGHTS


def test_cc_by_nc_weights_require_explicit_opt_in(monkeypatch):
    monkeypatch.delenv(WEIGHTS_ENV_VAR, raising=False)
    # Never selected by default...
    assert resolve_weights(MappingOptions()) != CC_BY_NC_WEIGHTS
    # ...only via an explicit option...
    assert resolve_weights(MappingOptions(extra={"weights": CC_BY_NC_WEIGHTS})) == CC_BY_NC_WEIGHTS
    # ...or the explicit env var.
    monkeypatch.setenv(WEIGHTS_ENV_VAR, CC_BY_NC_WEIGHTS)
    assert resolve_weights(None) == CC_BY_NC_WEIGHTS


def test_explicit_option_beats_env(monkeypatch):
    monkeypatch.setenv(WEIGHTS_ENV_VAR, CC_BY_NC_WEIGHTS)
    assert resolve_weights(MappingOptions(extra={"weights": APACHE_WEIGHTS})) == APACHE_WEIGHTS


def test_selected_weights_and_license_recorded_in_stats(monkeypatch):
    _install_mock(monkeypatch)
    backend = MapAnythingBackend()

    result = backend.map(
        make_synthetic_views(2),
        correspondences=None,
        options=MappingOptions(extra={"weights": CC_BY_NC_WEIGHTS}),
    )

    assert result.stats["weights"] == CC_BY_NC_WEIGHTS
    assert result.stats["weights_license"] == "CC-BY-NC-4.0"


# --- engine-gated real inference (skips cleanly without the engine) --------


@pytest.mark.needs_mapanything
def test_real_engine_backend_is_a_conforming_mapper():
    """Engine-gated: with torch + mapanything installed the real path conforms."""
    pytest.importorskip("torch")
    pytest.importorskip("mapanything")

    backend = MapAnythingBackend()
    assert isinstance(backend, Mapper)
    # A real 2-view inference over tiny synthetic images.
    result = backend.map(make_synthetic_views(2, height=224, width=224), correspondences=None)
    assert isinstance(result, MappingResult)
    assert len(result.poses) == 2
