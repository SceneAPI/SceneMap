"""``MapAnythingBackend`` — the feed-forward :class:`sceneio.mapping.Mapper`
conformer for Meta + CMU's MapAnything (arXiv 2509.13414).

This is the *feed-forward* family: a single transformer directly regresses
factored metric 3-D geometry (per-view depth/ray maps, camera poses, a
metric scale factor) from raw views — **no correspondences required**
(``requires_correspondences=False``). It optionally consumes calibration,
pose, and depth priors, and emits dense per-view pointmaps + per-pixel
confidence. It is the proof point that a learned mapper plugs into the
neutral sceneio contract with no core routing changes: core's
``io_mapper()`` resolver structurally accepts this object because it
satisfies the ``Mapper`` protocol (``traits()`` + ``map()``) and its
``traits()`` returns ``MapperTraits``.

The module owns three translations the contract needs:

1. **ViewInput -> model view dict** (:func:`build_view_payload`): the image
   becomes ``img`` (H, W, 3); an optional :class:`~sceneio.data.Calibration`
   becomes ``intrinsics`` (a 3x3 K) XOR ``ray_directions`` (H, W, 3); an
   optional :class:`~sceneio.data.PosePrior` becomes ``camera_poses``
   (4x4 cam2world OpenCV) + ``is_metric_scale``; an optional
   :class:`~sceneio.data.DepthMap` prior becomes ``depth_z``.
2. **the model call** (:func:`_run_inference`): the SINGLE lazy-imported
   entry point that touches torch / mapanything. Everything uncertain
   about the real inference API is contained here; tests monkeypatch it
   with canned numpy predictions so the rest of the adapter (input build,
   output conversion, fusion, frame logic) is exercised with no engine.
3. **predictions -> MappingResult** (:func:`result_from_predictions`):
   per-view ``camera_poses`` -> cam2world :class:`~sceneio.data.SE3`;
   predicted ``intrinsics`` -> per-view :class:`~sceneio.data.Calibration`;
   ``pts3d`` + ``conf`` -> the dense
   ``(Pointmap, ConfidenceMap)`` payload; the per-view pointmaps fused into a
   subsampled :class:`~sceneio.data.TrackedPointCloud`.

Weights/license (locked decision): the default is the Apache-2.0 variant
``facebook/map-anything-apache``; the better CC-BY-NC-4.0 variant
``facebook/map-anything`` is an explicit opt-in (a ``MappingOptions.extra``
``"weights"`` key or the ``SCENEAPI_MAPANYTHING_WEIGHTS`` env var). The
non-commercial weights are NEVER the default — see :func:`resolve_weights`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np
from sceneio.data import (
    SE3,
    Calibration,
    CameraIntrinsics,
    CameraModel,
    ConfidenceMap,
    CorrespondenceGraph,
    FrameMeta,
    Pointmap,
    TrackedPointCloud,
    ViewInput,
)
from sceneio.errors import ContractViolation
from sceneio.imagesource import MaterializedImage
from sceneio.mapping import MapperTraits, MappingOptions, MappingResult

try:
    from sceneapi.errors import CapabilityUnavailableError
except ModuleNotFoundError:  # pragma: no cover - only for package metadata tools

    class CapabilityUnavailableError(RuntimeError):  # type: ignore[no-redef]
        def __init__(self, *, capability: str, reason: str = "") -> None:
            super().__init__(reason or capability)


# --- weights / license (locked decision) -----------------------------------

# Apache-2.0 weights: the SAFE default (commercial-use OK).
APACHE_WEIGHTS = "facebook/map-anything-apache"
# CC-BY-NC-4.0 weights: better, but NON-COMMERCIAL — opt-in only, never default.
CC_BY_NC_WEIGHTS = "facebook/map-anything"
DEFAULT_WEIGHTS = APACHE_WEIGHTS
WEIGHTS_ENV_VAR = "SCENEAPI_MAPANYTHING_WEIGHTS"

# Known Hugging Face weight ids -> their license (for honest runtime metadata).
KNOWN_WEIGHTS_LICENSES: dict[str, str] = {
    APACHE_WEIGHTS: "Apache-2.0",
    CC_BY_NC_WEIGHTS: "CC-BY-NC-4.0",
}


def resolve_weights(options: MappingOptions | None = None) -> str:
    """Resolve the Hugging Face weights id, defaulting to the Apache variant.

    Precedence: an explicit ``options.extra["weights"]`` beats the
    ``SCENEAPI_MAPANYTHING_WEIGHTS`` env var, which beats the default
    :data:`APACHE_WEIGHTS`. The CC-BY-NC-4.0 weights (:data:`CC_BY_NC_WEIGHTS`)
    are therefore reachable ONLY when a caller/operator names them
    explicitly — they are never selected by default. Selecting them opts
    into the upstream non-commercial license term for whoever runs the
    model; the wrapper does not add or remove that term.
    """
    requested: object = None
    if options is not None:
        requested = options.extra.get("weights")
    if requested is None:
        requested = os.environ.get(WEIGHTS_ENV_VAR)
    if requested is None or requested == "":
        return DEFAULT_WEIGHTS
    return str(requested)


def weights_license(weights: str) -> str:
    """The license of a known weights id, or ``"unknown"`` for custom ids."""
    return KNOWN_WEIGHTS_LICENSES.get(weights, "unknown")


# --- ViewInput -> model view dict ------------------------------------------


def _materialize_image(image: Any) -> np.ndarray:
    """An (H, W, 3) uint8 RGB array for MapAnything's ``img`` input.

    In-memory arrays are used directly (gray is expanded to 3 channels);
    a :class:`MaterializedImage` reference is loaded from disk with Pillow
    (lazily imported — only the real-inference path touches pixels on disk).
    """
    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        return np.ascontiguousarray(array.astype(np.uint8, copy=False))
    if isinstance(image, MaterializedImage):
        from PIL import Image as PILImage

        with PILImage.open(image.abs_path) as handle:
            return np.ascontiguousarray(np.asarray(handle.convert("RGB"), dtype=np.uint8))
    raise ContractViolation(
        f"MapAnythingBackend: cannot materialize image of type {type(image).__name__}"
    )


def _intrinsics_matrix(intrinsics: CameraIntrinsics) -> np.ndarray:
    """A (3, 3) pinhole camera matrix from COLMAP-model params.

    MapAnything's ``intrinsics`` input is a plain pinhole K; distortion
    params (if any) are dropped — the model recovers its own calibration
    and the prior only seeds focal length + principal point.
    """
    names = intrinsics.model.param_names
    values = {name: float(param) for name, param in zip(names, intrinsics.params, strict=True)}
    if "fx" in values and "fy" in values:
        fx, fy = values["fx"], values["fy"]
    else:  # SIMPLE_* models share a single focal length ``f``.
        fx = fy = values["f"]
    cx = values.get("cx", intrinsics.width / 2.0)
    cy = values.get("cy", intrinsics.height / 2.0)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def build_view_payload(view: ViewInput, traits: MapperTraits) -> dict[str, Any]:
    """One MapAnything input view dict built from a :class:`ViewInput`.

    Only the priors the traits accept are attached, so the payload never
    silently promotes an optional input to required. ``depth_z`` is only
    attached alongside a calibration (the model requires calibration to
    consume metric depth).
    """
    payload: dict[str, Any] = {"img": _materialize_image(view.image)}

    calibrated = False
    if traits.accepts_calibration and view.calibration is not None:
        if view.calibration.intrinsics is not None:
            payload["intrinsics"] = _intrinsics_matrix(view.calibration.intrinsics)
            calibrated = True
        elif view.calibration.rays is not None:
            payload["ray_directions"] = np.asarray(
                view.calibration.rays.directions, dtype=np.float32
            )
            calibrated = True

    if traits.accepts_pose_priors and view.pose_prior is not None:
        pose = view.pose_prior.pose.as_convention("opencv_cam2world")
        payload["camera_poses"] = np.asarray(pose.matrix, dtype=np.float64)
        if view.pose_prior.is_metric:
            payload["is_metric_scale"] = True

    if (
        traits.accepts_depth_priors
        and view.depth_prior is not None
        and calibrated  # MapAnything: depth_z requires calibration.
    ):
        payload["depth_z"] = np.asarray(view.depth_prior.depth, dtype=np.float32)

    return payload


# --- the model call (the ONE lazy-imported entry point) --------------------

# Loaded models are cached per weights id so repeated map() calls in one
# process reuse the (expensive) load.
_MODEL_CACHE: dict[str, Any] = {}


def _load_model(weights: str) -> tuple[Any, str]:  # pragma: no cover - needs torch + weights
    """Load + cache the MapAnything model for ``weights`` (lazy torch import)."""
    cached = _MODEL_CACHE.get(weights)
    if cached is not None:
        return cached
    import torch
    from mapanything.models import MapAnything

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MapAnything.from_pretrained(weights).to(device)
    model.eval()
    _MODEL_CACHE[weights] = (model, device)
    return model, device


def _prediction_to_numpy(pred: Any) -> dict[str, np.ndarray]:  # pragma: no cover - needs torch
    """Extract a per-view MapAnything prediction into plain numpy arrays.

    Leading singleton batch dims are squeezed so callers see per-view
    ``(H, W, 3)`` / ``(H, W)`` / ``(3, 3)`` / ``(4, 4)`` shapes regardless of
    whether the model returned batched tensors.
    """

    def _to_numpy(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().to("cpu").float().numpy()
        return np.asarray(value)

    def _drop_batch(array: np.ndarray | None, target_ndim: int) -> np.ndarray | None:
        if array is None:
            return None
        while array.ndim > target_ndim and array.shape[0] == 1:
            array = array[0]
        return array

    getter = pred.get if isinstance(pred, dict) else (lambda key: getattr(pred, key, None))
    out: dict[str, np.ndarray] = {}
    for key, target in (
        ("pts3d", 3),
        ("conf", 2),
        ("intrinsics", 2),
        ("camera_poses", 2),
        ("img_no_norm", 3),
    ):
        array = _drop_batch(_to_numpy(getter(key)), target)
        if array is not None:
            out[key] = array
    mask = _drop_batch(_to_numpy(getter("mask")), 2)
    if mask is not None:
        out["mask"] = mask.astype(bool)
    return out


def _run_inference(
    payloads: Sequence[dict[str, Any]],
    *,
    weights: str,
    options: MappingOptions,
) -> list[dict[str, np.ndarray]]:  # pragma: no cover - the engine-gated real call
    """THE single real-model call: numpy view payloads -> numpy predictions.

    This is the ONLY function that imports torch / mapanything and runs
    ``model.infer``. It is deliberately isolated so the (heavy, engine-gated)
    model call is the single point of uncertainty about the upstream API —
    tests monkeypatch this function with canned numpy predictions, exercising
    the whole adapter around it without an engine.

    Returns one prediction dict per input view with numpy arrays keyed
    ``pts3d`` (H, W, 3 world points), ``conf`` (H, W), ``intrinsics`` (3, 3),
    ``camera_poses`` (4, 4 cam2world), and optionally ``img_no_norm`` /
    ``mask``.

    NOTE (contained uncertainty): MapAnything's own ``load_images`` helper
    resizes/normalizes inputs to model-friendly dimensions. This function
    passes ``img`` at native resolution; if an installed build requires the
    upstream preprocessing, apply ``mapanything.utils.image`` here — this is
    the single spot to reconcile against real weights.
    """
    import torch

    model, device = _load_model(weights)

    views: list[dict[str, Any]] = []
    for payload in payloads:
        view: dict[str, Any] = {
            "img": torch.from_numpy(np.ascontiguousarray(payload["img"])).to(device)
        }
        for key in ("intrinsics", "ray_directions", "depth_z", "camera_poses"):
            if key in payload:
                view[key] = torch.from_numpy(
                    np.ascontiguousarray(np.asarray(payload[key], dtype=np.float32))
                ).to(device)
        if payload.get("is_metric_scale"):
            view["is_metric_scale"] = torch.tensor([True], device=device)
        views.append(view)

    infer_kwargs: dict[str, Any] = {
        "memory_efficient_inference": True,
        "use_amp": True,
        "amp_dtype": "bf16",
    }
    extra_kwargs = options.extra.get("infer_kwargs")
    if isinstance(extra_kwargs, dict):
        infer_kwargs.update(extra_kwargs)

    with torch.no_grad():
        predictions = model.infer(views, **infer_kwargs)
    return [_prediction_to_numpy(pred) for pred in predictions]


# --- predictions -> MappingResult ------------------------------------------


def _orthonormalize(rotation: np.ndarray) -> np.ndarray:
    """Nearest proper rotation (SVD polar), taming model numerical drift.

    The SE3 contract validates orthonormality to 1e-5; a raw model pose can
    drift past that, so we project onto SO(3) before constructing the SE3.
    """
    u, _, vt = np.linalg.svd(rotation.astype(np.float64, copy=False))
    proper = u @ vt
    if np.linalg.det(proper) < 0:  # reflection -> flip the smallest singular axis
        u[:, -1] *= -1
        proper = u @ vt
    return proper


def _se3_from_cam2world(matrix: np.ndarray) -> SE3:
    """A cam2world OpenCV :class:`SE3` from a (4, 4) model pose matrix."""
    matrix = np.asarray(matrix, dtype=np.float64)
    return SE3(
        rotation=_orthonormalize(matrix[:3, :3]),
        translation=matrix[:3, 3],
        convention="opencv_cam2world",
    )


def _calibration_from_k(matrix: np.ndarray, height: int, width: int) -> Calibration:
    """A PINHOLE :class:`Calibration` from a predicted (3, 3) K matrix."""
    matrix = np.asarray(matrix, dtype=np.float64)
    params = np.array([matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]], dtype=np.float64)
    return Calibration.from_intrinsics(
        CameraIntrinsics(
            model=CameraModel.PINHOLE, width=int(width), height=int(height), params=params
        )
    )


def _confidence_map(conf: np.ndarray) -> ConfidenceMap:
    """A [0, 1] :class:`ConfidenceMap` from MapAnything's ``conf`` scores.

    MapAnything's confidence is a non-negative score that can exceed 1, while
    :class:`ConfidenceMap` requires finite values in [0, 1]. Non-finite
    entries are zeroed and, when the per-view max exceeds 1, the field is
    normalized by that max — a monotonic, per-view rescale (relative
    confidence within the view is preserved; absolute magnitude is not).
    """
    values = np.asarray(conf, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    peak = float(values.max()) if values.size else 0.0
    if peak > 1.0:
        values = values / peak
    values = np.clip(values, 0.0, 1.0).astype(np.float32)
    return ConfidenceMap(values=values)


def _view_hw(pred: dict[str, np.ndarray], view: ViewInput) -> tuple[int, int]:
    """(H, W) for a view, from its prediction arrays or the input image."""
    for key in ("pts3d", "conf", "img_no_norm"):
        array = pred.get(key)
        if array is not None and array.ndim >= 2:
            return int(array.shape[0]), int(array.shape[1])
    image = view.image
    if isinstance(image, np.ndarray):
        return int(image.shape[0]), int(image.shape[1])
    if view.calibration is not None:
        return view.calibration.image_size
    raise ContractViolation("MapAnythingBackend: cannot determine view resolution")


def _fuse_point_cloud(
    preds: Sequence[dict[str, np.ndarray]], options: MappingOptions
) -> TrackedPointCloud | None:
    """Fuse per-view world-frame pointmaps into a subsampled cloud.

    Design decision (documented, in scope): the per-view ``pts3d`` (already
    in the shared world frame) are concatenated, filtered to finite +
    in-``mask`` pixels, coloured from ``img_no_norm`` when every contributing
    view carries colour, and subsampled to a cap (``options.extra["max_points"]``,
    default 200k) so a huge multi-view run still yields a servable sparse
    cloud. Feed-forward mapping has no correspondences, so there are no track
    observations. Returns ``None`` when no finite point survives.
    """
    cap = int(options.extra.get("max_points", 200_000))
    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray | None] = []
    for pred in preds:
        pts = pred.get("pts3d")
        if pts is None:
            continue
        points = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        valid = np.isfinite(points).all(axis=1)
        mask = pred.get("mask")
        if mask is not None:
            valid &= np.asarray(mask, dtype=bool).reshape(-1)
        points = points[valid]
        if points.size == 0:
            continue
        xyz_parts.append(points)
        color = pred.get("img_no_norm")
        if color is None:
            rgb_parts.append(None)
        else:
            rgb_parts.append(
                np.clip(np.asarray(color).reshape(-1, 3)[valid], 0, 255).astype(np.uint8)
            )

    if not xyz_parts:
        return None
    xyz = np.concatenate(xyz_parts, axis=0)
    rgb: np.ndarray | None = None
    if all(part is not None for part in rgb_parts):
        rgb = np.concatenate([part for part in rgb_parts if part is not None], axis=0)

    if xyz.shape[0] > cap:
        rng = np.random.default_rng(options.seed if options.seed is not None else 0)
        keep = rng.choice(xyz.shape[0], size=cap, replace=False)
        keep.sort()
        xyz = xyz[keep]
        if rgb is not None:
            rgb = rgb[keep]
    return TrackedPointCloud(
        xyz=np.ascontiguousarray(xyz), rgb=None if rgb is None else np.ascontiguousarray(rgb)
    )


def frame_meta(views: Sequence[ViewInput], *, weights: str, options: MappingOptions) -> FrameMeta:
    """Declare the output frame + honest scale claim.

    ``world_frame="first_view"`` is the learned-family convention (the world
    frame is anchored at the first view's camera). Scale honesty (a judgment
    call mirroring the ColmapMapper precedent of claiming metric ONLY when
    anchored):

    - any input view with a metric :class:`~sceneio.data.PosePrior`
      (``is_metric=True``) anchors real units -> ``scale="metric"``,
      ``scale_provenance="prior_anchored"``.
    - otherwise the model's own (unanchored) metric prediction is reported as
      ``scale="normalized"`` with ``scale_provenance="model_claimed"`` — a
      model's say-so is weaker evidence than a metric prior. An explicit
      opt-in (``options.extra["trust_model_metric"]=True``) upgrades that to a
      ``model_claimed`` metric claim when the weights are metric-capable.
    """
    metric_prior = any(view.pose_prior is not None and view.pose_prior.is_metric for view in views)
    if metric_prior:
        return FrameMeta(
            world_frame="first_view", scale="metric", scale_provenance="prior_anchored"
        )
    if bool(options.extra.get("trust_model_metric", False)):
        return FrameMeta(world_frame="first_view", scale="metric", scale_provenance="model_claimed")
    return FrameMeta(world_frame="first_view", scale="normalized", scale_provenance="model_claimed")


def result_from_predictions(
    views: Sequence[ViewInput],
    preds: Sequence[dict[str, np.ndarray]],
    *,
    weights: str,
    options: MappingOptions,
) -> MappingResult:
    """Convert per-view MapAnything predictions into a view-aligned result.

    Feed-forward registers every view, so the happy path has a pose per
    view and no ``None`` slots — but the conversion stays ``None``-tolerant
    (a prediction missing ``camera_poses`` yields an unregistered view with
    ``None`` pose / calibration / dense entry) so a partial run still
    produces a valid, index-aligned result.
    """
    view_list = list(views)
    poses: list[SE3 | None] = []
    calibrations: list[Calibration | None] = []
    dense: list[tuple[Pointmap, ConfidenceMap] | None] = []

    for index, view in enumerate(view_list):
        pred = preds[index] if index < len(preds) else {}
        pose_matrix = pred.get("camera_poses")
        if pose_matrix is None:
            poses.append(None)
            calibrations.append(None)
            dense.append(None)
            continue
        poses.append(_se3_from_cam2world(pose_matrix))
        height, width = _view_hw(pred, view)
        k_matrix = pred.get("intrinsics")
        calibrations.append(
            _calibration_from_k(k_matrix, height, width) if k_matrix is not None else None
        )
        pts = pred.get("pts3d")
        conf = pred.get("conf")
        if pts is not None and conf is not None:
            dense.append(
                (
                    Pointmap(points=np.asarray(pts, dtype=np.float32), frame="world"),
                    _confidence_map(conf),
                )
            )
        else:
            dense.append(None)

    have_calibration = any(calibration is not None for calibration in calibrations)
    geometry = _fuse_point_cloud(preds, options)
    stats: dict[str, Any] = {
        "engine": "mapanything.infer",
        "weights": weights,
        "weights_license": weights_license(weights),
        "num_input_views": len(view_list),
        "num_registered": sum(1 for pose in poses if pose is not None),
    }
    if geometry is not None:
        stats["num_fused_points"] = len(geometry)
    return MappingResult(
        poses=tuple(poses),
        frame=frame_meta(view_list, weights=weights, options=options),
        calibrations=tuple(calibrations) if have_calibration else None,
        geometry=geometry,
        dense=tuple(dense),
        stats=stats,
    )


# --- the backend ------------------------------------------------------------


class MapAnythingBackend:
    """MapAnything feed-forward provider — a neutral sceneio ``Mapper``.

    Implements the minimum sceneapi Backend identity (``name`` / ``version`` /
    ``vendor`` / :meth:`capabilities` / :meth:`runtime_versions`) AND the
    sceneio ``Mapper`` contract (:meth:`traits` / :meth:`map`), so core's
    dual-dispatch ``io_mapper()`` resolver routes feed-forward mapping to it
    with no core changes.
    """

    name = "mapanything"
    version = "0.1.0"
    vendor = "Meta Reality Labs + CMU"

    def capabilities(self) -> set[str]:
        """The one portable capability, keyed on the Mapper contract.

        Mirrors the core StubBackend's feed-forward advertisement: the
        capability reflects the feed-forward mapping CONTRACT this object
        implements (the same io-Mapper presence core's dual-dispatch keys
        on), not whether torch + weights are provisioned. Actual inference is
        engine-gated — :meth:`map` raises at run time when the runtime is
        absent.
        """
        from sceneio.mapping import Mapper

        caps: set[str] = set()
        if isinstance(self, Mapper):
            traits = self.traits()
            if isinstance(traits, MapperTraits) and not traits.requires_correspondences:
                caps.add("map.feed_forward")
        return caps

    def traits(self) -> MapperTraits:
        return MapperTraits(
            # Feed-forward: raw views in, no correspondences needed.
            requires_correspondences=False,
            # Consumes pose priors (camera_poses) + metric anchor (is_metric_scale).
            accepts_pose_priors=True,
            # Consumes depth priors (depth_z, alongside calibration).
            accepts_depth_priors=True,
            # Consumes calibration as intrinsics OR ray_directions.
            accepts_calibration=True,
            # Emits per-view (Pointmap, ConfidenceMap).
            emits_dense=True,
            # Metric-capable (native metric model); the per-run claim is honest,
            # see frame_meta — metric only when anchored.
            metric_capable=True,
        )

    def map(
        self,
        views: Sequence[ViewInput],
        *,
        correspondences: CorrespondenceGraph | None = None,
        options: MappingOptions | None = None,
    ) -> MappingResult:
        """Run feed-forward mapping over ``views``.

        Honors ``requires_correspondences=False``: ``correspondences`` is
        accepted whether ``None`` or a graph, and always ignored (feed-forward
        needs none) — it never errors on either. ``MappingOptions.max_views``
        is not used to drop views: the result stays index-aligned to every
        input view (dropping would break that alignment); pass fewer views to
        map fewer.
        """
        # `correspondences` intentionally ignored — feed-forward needs none.
        del correspondences
        options = options or MappingOptions()
        view_list = list(views)
        if not view_list:
            raise ContractViolation("MapAnythingBackend.map: at least one view is required")

        traits = self.traits()
        weights = resolve_weights(options)
        payloads = [build_view_payload(view, traits) for view in view_list]
        preds = _run_inference(payloads, weights=weights, options=options)
        return result_from_predictions(view_list, preds, weights=weights, options=options)

    def runtime_versions(self) -> dict[str, str]:
        """Opaque version salt for core's cache key (torch/mapanything probed)."""
        versions: dict[str, str] = {
            "backend": self.version,
            "default_weights": DEFAULT_WEIGHTS,
        }
        versions.update(_engine_runtime_versions())
        return versions


def _engine_runtime_versions() -> dict[str, str]:
    """Best-effort torch + mapanything version probe (never raises)."""
    out: dict[str, str] = {}
    try:
        import torch

        out["torch_version"] = str(torch.__version__)
        out["torch_cuda"] = str(torch.version.cuda or "cpu")
    except Exception as exc:  # torch is provisioning-installed; absent in CI
        out["torch_status"] = f"unavailable: {type(exc).__name__}"
    try:
        import mapanything

        out["mapanything_version"] = str(getattr(mapanything, "__version__", "unknown"))
    except Exception as exc:  # git-only engine; absent in CI
        out["mapanything_status"] = f"unavailable: {type(exc).__name__}"
    return out


def backend_factory() -> MapAnythingBackend:
    return MapAnythingBackend()


__all__ = [
    "APACHE_WEIGHTS",
    "CC_BY_NC_WEIGHTS",
    "DEFAULT_WEIGHTS",
    "WEIGHTS_ENV_VAR",
    "MapAnythingBackend",
    "backend_factory",
    "build_view_payload",
    "frame_meta",
    "resolve_weights",
    "result_from_predictions",
    "weights_license",
]
