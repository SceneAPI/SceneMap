from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

try:
    from sceneapi.errors import CapabilityUnavailableError, NotFoundError, ValidationError
except ModuleNotFoundError:  # pragma: no cover - allows adapter tests without sfmapi installed

    class CapabilityUnavailableError(RuntimeError):  # type: ignore[no-redef]
        def __init__(self, *, capability: str, reason: str = "") -> None:
            super().__init__(reason or capability)

    class NotFoundError(RuntimeError):  # type: ignore[no-redef]
        pass

    class ValidationError(RuntimeError):  # type: ignore[no-redef]
        pass


# parents[3]: backend.py lives one package level deeper than in the
# superseded per-family repo (src/scenemap/<family>/backend.py).
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPHERESFM_ROOT = REPO_ROOT / "third_party" / "spheresfm"

SPHERESFM_COMMANDS: tuple[str, ...] = (
    "help",
    "automatic_reconstructor",
    "bundle_adjuster",
    "color_extractor",
    "database_cleaner",
    "database_creator",
    "database_merger",
    "delaunay_mesher",
    "exhaustive_matcher",
    "feature_extractor",
    "feature_importer",
    "hierarchical_mapper",
    "image_deleter",
    "image_filterer",
    "image_rectifier",
    "image_registrator",
    "image_undistorter",
    "image_undistorter_standalone",
    "mapper",
    "matches_importer",
    "model_aligner",
    "model_analyzer",
    "model_comparer",
    "model_converter",
    "model_cropper",
    "model_merger",
    "model_orientation_aligner",
    "model_splitter",
    "model_transformer",
    "patch_match_stereo",
    "point_filtering",
    "point_triangulator",
    "poisson_mesher",
    "project_generator",
    "rig_bundle_adjuster",
    "sequential_matcher",
    "spatial_matcher",
    "sphere_cubic_reprojecer",
    "stereo_fusion",
    "transitive_matcher",
    "vocab_tree_builder",
    "vocab_tree_matcher",
    "vocab_tree_retriever",
)
SPHERESFM_COMMAND_SET = frozenset(SPHERESFM_COMMANDS)
READ_ONLY_COMMANDS = {"help", "model_analyzer", "model_comparer"}
MATCHING_MODES = {"spatial", "vocabtree", "exhaustive", "sequential"}
_STAGE_METADATA_KEYS = {
    "backend_options",
    "input_artifacts",
    "portable",
    "provider",
    "max_num_features",
    "seed",
    "sift",
    "strategy",
    "type",
    "use_gpu",
    "version",
}

# Default SPHERE camera parameters used by SphereSfM's documented panorama
# pipeline (focal scale + principal point for a 2:1 equirectangular image).
DEFAULT_SPHERE_CAMERA_PARAMS = "1,3520,1760"

# Portable sfmapi pair-selection capability -> SphereSfM matcher command.
# SphereSfM is a COLMAP fork: each matcher runs geometric verification
# inline and writes ``two_view_geometries`` directly, so there is no
# separate ``matches.verify`` stage to wrap.
PAIR_MODE_COMMANDS: dict[str, str] = {
    "exhaustive": "exhaustive_matcher",
    "sequential": "sequential_matcher",
    "spatial": "spatial_matcher",
    "vocabtree": "vocab_tree_matcher",
}

# Portable sfmapi pair-selection capability -> matcher command, mirrored
# from ``PAIR_MODE_COMMANDS``. Each matcher mode SphereSfM dispatches in
# ``match()`` advertises its own ``pairs.*`` capability string.
PAIR_MODE_CAPABILITIES: dict[str, str] = {
    "exhaustive": "pairs.exhaustive",
    "sequential": "pairs.sequential",
    "spatial": "pairs.spatial",
    "vocabtree": "pairs.vocabtree",
}

# Portable sfmapi export format -> ``model_converter --output_type`` value.
EXPORT_FORMAT_OUTPUT_TYPES: dict[str, str] = {
    "ply": "PLY",
    "colmap_text": "TXT",
    "colmap_bin": "BIN",
    "nvm": "NVM",
}

# Portable sfmapi capabilities this backend implements as thin wrappers
# over the SphereSfM (COLMAP-fork) CLI. Backend-native COLMAP verbs stay
# in the action catalog and are intentionally absent here. Every name
# below resolves to a real wrapper method on :class:`SphereSfMBackend`.
SPHERESFM_CAPABILITIES: frozenset[str] = frozenset(
    {
        "features.extract.sift",
        # match() dispatches all four COLMAP-fork matchers via
        # PAIR_MODE_COMMANDS; each pair-selection strategy is portable.
        "pairs.exhaustive",
        "pairs.sequential",
        "pairs.spatial",
        "pairs.vocabtree",
        # mapper / hierarchical_mapper cover all three sparse mapping kinds
        # plus the SPHERE-camera spherical variant.
        "map.incremental",
        "map.hierarchical",
        "map.spherical",
        # bundle_adjuster / rig_bundle_adjuster / point_triangulator /
        # image_registrator refinement wrappers.
        "ba.standard",
        "ba.rig",
        "triangulate.retri",
        "relocalize.images",
        # model_merger pairwise fold-left over N sub-models.
        "recon.merge",
        # model_converter --output_type {PLY,TXT,BIN,NVM}.
        "export.ply",
        "export.colmap_text",
        "export.colmap_bin",
        "export.nvm",
        # model_aligner / model_transformer Sim(3) application.
        "georegister.sim3",
        "projection.cubemap_rig",
    }
)


def _expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value).strip().strip('"'))).expanduser()


def _plugin_cache_root(plugin_id: str) -> Path:
    override = os.environ.get("SFMAPI_PLUGIN_CACHE")
    if override:
        return _expand_path(override) / plugin_id
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "sfmapi" / "plugins" / plugin_id
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "sfmapi" / "plugins" / plugin_id


def _cache_executable_candidates() -> list[Path]:
    cache = _plugin_cache_root("spheresfm")
    names = ["colmap.exe", "colmap"] if os.name == "nt" else ["colmap", "colmap.exe"]
    candidates = [cache / "current" / name for name in names]
    if cache.exists():
        for child in cache.iterdir():
            if child.is_dir() and child.name != "current":
                candidates.extend(child / name for name in names)
    return candidates


def _cli_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """Drop sfmapi routing/envelope fields before building CLI flags."""
    return {
        key: value for key, value in dict(options or {}).items() if key not in _STAGE_METADATA_KEYS
    }


def _default_executable_candidates() -> list[Path]:
    names = ["colmap.exe", "colmap"] if os.name == "nt" else ["colmap", "colmap.exe"]
    relative_dirs = [
        Path("build") / "src" / "exe" / "Release",
        Path("build") / "src" / "exe" / "Debug",
        Path("build") / "src" / "exe",
        Path("build") / "src" / "exe" / "RelWithDebInfo",
    ]
    return [
        DEFAULT_SPHERESFM_ROOT / relative_dir / name
        for relative_dir in relative_dirs
        for name in names
    ] + _cache_executable_candidates()


def resolve_spheresfm_executable(value: str | Path | None) -> Path | None:
    raw = value or os.environ.get("SFMAPI_SPHERESFM_EXECUTABLE")
    if raw:
        path = _expand_path(raw)
        candidates = [path]
        if path.is_dir():
            candidates = [
                path / "colmap.exe",
                path / "colmap",
                path / "bin" / "colmap.exe",
                path / "bin" / "colmap",
            ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None

    for candidate in _default_executable_candidates():
        if candidate.exists():
            return candidate.resolve()
    return None


def configure_spheresfm_environment(
    executable: str | Path | None = None,
    *,
    validate: bool = False,
) -> Path | None:
    resolved = resolve_spheresfm_executable(executable)
    if resolved is None:
        if validate:
            raise ValueError(
                "SphereSfM executable not found. Build the upstream submodule and set "
                "SFMAPI_SPHERESFM_EXECUTABLE or pass --spheresfm-executable."
            )
        return None
    os.environ["SFMAPI_SPHERESFM_EXECUTABLE"] = str(resolved)
    existing = os.environ.get("PATH", "")
    parent = str(resolved.parent)
    if parent not in existing.split(os.pathsep):
        os.environ["PATH"] = os.pathsep.join([parent, existing])
    return resolved


class SphereSfMBackend:
    name = "spheresfm"
    version = "0.0.1"
    vendor = "SphereSfM"

    def __init__(self, executable: str | Path | None = None) -> None:
        self._executable_override = _expand_path(executable).resolve() if executable else None

    def capabilities(self) -> set[str]:
        """Portable sfmapi capabilities backed by real wrapper methods.

        SphereSfM is a COLMAP fork, so the full sparse pipeline —
        feature extraction, every pair-selection matcher, incremental /
        hierarchical / spherical mapping, bundle adjustment (incl. rig),
        re-triangulation, image registration, model merging, export,
        and Sim(3) georegistration — is wired to the corresponding
        ``colmap`` sub-commands. The set is empty until the SphereSfM
        executable is resolvable (mirrors ``sfmapi_colmap_cli``): a
        capability the deployment cannot actually run must not be
        advertised.
        """
        if self._find_executable() is None:
            return set()
        return set(SPHERESFM_CAPABILITIES)

    def runtime_versions(self) -> dict[str, str]:
        executable = self._find_executable()
        versions = {
            "backend": self.version,
            "spheresfm_root": str(DEFAULT_SPHERESFM_ROOT),
            "spheresfm_executable": str(executable) if executable else "missing",
        }
        commit = self._git_revision(DEFAULT_SPHERESFM_ROOT)
        if commit:
            versions["spheresfm_commit"] = commit
        return versions

    # ------------------------------------------------------------------
    # Portable sfmapi stage protocols.
    #
    # These are thin wrappers over the same ``_run_colmap`` calls the
    # ``spheresfm.reconstructPanoramaFolder`` action already issues. They
    # exist so the portable feature/mapping/projection capabilities the
    # plugin manifest advertises resolve to real code instead of 501s.
    # ------------------------------------------------------------------

    def extract_features(
        self,
        *,
        database_path: Path,
        image_root: Path,
        image_list: list[str],
        options: dict[str, Any],
        progress: Any | None = None,
    ) -> dict[str, Any]:
        """Run SphereSfM ``feature_extractor`` with the SPHERE camera model."""
        self._require_executable()
        database_path = Path(database_path)
        image_root = Path(image_root)
        database_path.parent.mkdir(parents=True, exist_ok=True)
        raw_options = dict(options or {})
        opts = _cli_options(raw_options)
        if raw_options.get("use_gpu") is not None:
            opts["SiftExtraction.use_gpu"] = int(bool(raw_options["use_gpu"]))
        if raw_options.get("max_num_features") is not None:
            opts["SiftExtraction.max_num_features"] = int(raw_options["max_num_features"])

        feature_options: dict[str, Any] = {
            "database_path": database_path,
            "image_path": image_root,
            "ImageReader.camera_model": "SPHERE",
            "ImageReader.camera_params": opts.pop("camera_params", DEFAULT_SPHERE_CAMERA_PARAMS),
            "ImageReader.single_camera": int(bool(opts.pop("single_camera", True))),
        }
        for passthrough in ("camera_mask_path", "pose_path"):
            value = opts.pop(passthrough, None)
            if value:
                feature_options[f"ImageReader.{passthrough}"] = value

        list_path: Path | None = None
        if image_list:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", suffix=".txt", delete=False
            ) as handle:
                list_path = Path(handle.name)
                for image in image_list:
                    handle.write(f"{image}\n")
            feature_options["image_list_path"] = list_path
        feature_options.update(opts)

        self._progress(progress, "feature_extraction", 0, 1)
        try:
            result = self._run_colmap("feature_extractor", options=feature_options)
        finally:
            if list_path is not None:
                list_path.unlink(missing_ok=True)
        self._progress(progress, "feature_extraction", 1, 1)
        return {
            "database_path": str(database_path),
            "num_images": len(image_list),
            "engine": "spheresfm feature_extractor",
            "command": result,
        }

    def match(
        self,
        *,
        database_path: Path,
        mode: str,
        options: dict[str, Any],
        progress: Any | None = None,
    ) -> dict[str, Any]:
        """Run a SphereSfM matcher for the requested pair-selection mode."""
        normalized = str(mode).replace("-", "_").lower()
        command = PAIR_MODE_COMMANDS.get(normalized)
        if command is None:
            raise CapabilityUnavailableError(
                capability="pairs.exhaustive",
                reason=(
                    "SphereSfM wraps exhaustive, sequential, spatial, and "
                    f"vocabtree pair selection; {mode!r} is not supported."
                ),
            )
        self._require_executable()
        match_options: dict[str, Any] = {"database_path": Path(database_path)}
        match_options.update(_cli_options(options))

        self._progress(progress, "matching", 0, 1)
        result = self._run_colmap(command, options=match_options)
        self._progress(progress, "matching", 1, 1)
        return {
            "database_path": str(database_path),
            "strategy": normalized,
            "engine": f"spheresfm {command}",
            "command": result,
        }

    def run_mapping(
        self,
        *,
        kind: str,
        db_path: Path,
        image_root: Path,
        sparse_root: Path,
        job_dir: Path,
        spec: dict[str, Any],
        pose_priors: dict[str, Any] | None = None,
        progress: Any | None = None,
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        """Run SphereSfM sparse mapping for the requested ``kind``.

        SphereSfM is a COLMAP fork, so every sparse mapping mode maps to
        a ``colmap`` sub-command:

        - ``incremental`` -> ``mapper`` (perspective cameras, no SPHERE flag)
        - ``spherical``   -> ``mapper`` with ``--Mapper.sphere_camera 1``
        - ``hierarchical``-> ``hierarchical_mapper``

        Returns ``(summaries, reconstructions)`` to satisfy the portable
        :class:`MappingBackend` protocol. SphereSfM emits COLMAP sparse
        model directories (``sparse/0`` ...); reading them back into a
        portable reconstruction object is the job of a
        :class:`ReconstructionReaderBackend`, which this CLI wrapper does
        not implement, so the reconstruction list is left empty and the
        summaries carry the on-disk ``model_path`` instead.
        """
        normalized = str(kind).replace("-", "_").lower()
        if normalized not in ("incremental", "spherical", "hierarchical"):
            raise CapabilityUnavailableError(
                capability=f"map.{kind}",
                reason=(
                    "SphereSfM implements portable incremental, spherical, and "
                    f"hierarchical mapping; {kind!r} is not supported."
                ),
            )
        self._require_executable()
        db_path = Path(db_path)
        image_root = Path(image_root)
        sparse_root = Path(sparse_root)
        job_dir = Path(job_dir)
        sparse_root.mkdir(parents=True, exist_ok=True)
        job_dir.mkdir(parents=True, exist_ok=True)

        command = "hierarchical_mapper" if normalized == "hierarchical" else "mapper"
        mapper_options: dict[str, Any] = {
            "database_path": db_path,
            "image_path": image_root,
            "output_path": sparse_root,
        }
        if normalized == "spherical":
            # SPHERE camera model: lock intrinsics, enable sphere mapping.
            mapper_options.update(
                {
                    "Mapper.ba_refine_focal_length": 0,
                    "Mapper.ba_refine_principal_point": 0,
                    "Mapper.ba_refine_extra_params": 0,
                    "Mapper.sphere_camera": 1,
                }
            )
        mapper_overrides = spec.get("mapper") or spec.get("options") or {}
        if isinstance(mapper_overrides, dict):
            mapper_options.update(mapper_overrides)

        phase = f"{normalized}_mapping"
        self._progress(progress, phase, 0, 1)
        result = self._run_colmap(command, options=mapper_options)
        self._progress(progress, phase, 1, 1)

        engine = f"spheresfm {command}"
        model_dirs = [path for path in sorted(sparse_root.iterdir()) if path.is_dir()]
        if not model_dirs and any(
            (sparse_root / name).exists()
            for name in ("cameras.bin", "cameras.txt", "images.bin", "images.txt")
        ):
            model_dirs = [sparse_root]
        summaries: list[dict[str, Any]] = []
        for model_dir in model_dirs:
            model_name = model_dir.name if model_dir != sparse_root else "0"
            summaries.append(
                {
                    "idx": int(model_name) if model_name.isdigit() else model_name,
                    "model_path": str(model_dir),
                    "engine": engine,
                }
            )
        if not summaries:
            summaries.append({"idx": 0, "model_path": str(sparse_root), "engine": engine})
        summaries[0].setdefault("command", result)
        return summaries, []

    def convert_spherical_to_cubemap(
        self,
        *,
        input_model_path: Path,
        input_image_path: Path,
        output_path: Path,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        """Reproject a spherical reconstruction into cubic perspective faces.

        Wraps SphereSfM's ``sphere_cubic_reprojecer``, which reads a
        spherical reconstruction and exports per-image cubic perspective
        views (``ExportPerspectiveCubic``). This is the
        ``projection.cubemap_rig`` capability — a reconstruction-driven
        rig export, not an image-only equirect transform.
        """
        self._require_executable()
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        self._progress(progress, "sphere_cubic_reprojecer", 0, 1)
        result = self._run_colmap(
            "sphere_cubic_reprojecer",
            options={
                "image_path": Path(input_image_path),
                "input_path": Path(input_model_path),
                "output_path": output_path,
            },
        )
        self._progress(progress, "sphere_cubic_reprojecer", 1, 1)
        return {
            "input_model_path": str(input_model_path),
            "output_path": str(output_path),
            "engine": "spheresfm sphere_cubic_reprojecer",
            "command": result,
        }

    # ------------------------------------------------------------------
    # Portable refinement / merge / export / transform stage protocols.
    # All thin ``_run_colmap`` wrappers over COLMAP-fork sub-commands
    # that already live in ``SPHERESFM_COMMANDS``.
    # ------------------------------------------------------------------

    def bundle_adjustment(
        self,
        *,
        model_path: Path,
        output_path: Path,
        spec: dict[str, Any],
        progress: Any | None = None,
    ) -> dict[str, Any]:
        """Run SphereSfM ``bundle_adjuster`` (or ``rig_bundle_adjuster``).

        ``spec["mode"] == "rig"`` selects the rig-aware bundle adjuster
        (capability ``ba.rig``); any other value runs the standard
        ``bundle_adjuster`` (capability ``ba.standard``). Extra COLMAP
        ``BundleAdjustment.*`` knobs ride through ``spec["options"]``.
        """
        self._require_executable()
        model_path = Path(model_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        spec = dict(spec or {})
        mode = str(spec.get("mode", "standard")).replace("-", "_").lower()
        command = "rig_bundle_adjuster" if mode == "rig" else "bundle_adjuster"
        ba_options: dict[str, Any] = {
            "input_path": model_path,
            "output_path": output_path,
        }
        rig_config_path = spec.get("rig_config_path") or spec.get("rig_config")
        if command == "rig_bundle_adjuster" and rig_config_path:
            ba_options["rig_config_path"] = rig_config_path
        overrides = spec.get("options") or spec.get("bundle_adjustment") or {}
        if isinstance(overrides, dict):
            ba_options.update(overrides)

        self._progress(progress, "bundle_adjustment", 0, 1)
        result = self._run_colmap(command, options=ba_options)
        self._progress(progress, "bundle_adjustment", 1, 1)
        return {
            "model_path": str(model_path),
            "output_path": str(output_path),
            "mode": "rig" if command == "rig_bundle_adjuster" else "standard",
            "engine": f"spheresfm {command}",
            "command": result,
        }

    def triangulate(
        self,
        *,
        model_path: Path,
        database_path: Path,
        image_root: Path,
        output_path: Path,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        """Re-triangulate an existing model with ``point_triangulator``.

        Capability ``triangulate.retri``: takes a model with known poses
        plus a populated COLMAP database and rebuilds the 3D points.
        """
        self._require_executable()
        model_path = Path(model_path)
        database_path = Path(database_path)
        image_root = Path(image_root)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self._progress(progress, "triangulation", 0, 1)
        result = self._run_colmap(
            "point_triangulator",
            options={
                "database_path": database_path,
                "image_path": image_root,
                "input_path": model_path,
                "output_path": output_path,
            },
        )
        self._progress(progress, "triangulation", 1, 1)
        return {
            "model_path": str(model_path),
            "output_path": str(output_path),
            "engine": "spheresfm point_triangulator",
            "command": result,
        }

    def relocalize(
        self,
        *,
        model_path: Path,
        database_path: Path,
        image_root: Path,
        output_path: Path,
        image_ids: list[int],
        progress: Any | None = None,
    ) -> dict[str, Any]:
        """Register additional images with ``image_registrator``.

        Capability ``relocalize.images``: COLMAP's ``image_registrator``
        registers every database image not already in the input model,
        so ``image_ids`` is advisory metadata only (the COLMAP-fork verb
        exposes no per-id selection flag).
        """
        self._require_executable()
        model_path = Path(model_path)
        database_path = Path(database_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self._progress(progress, "relocalization", 0, 1)
        result = self._run_colmap(
            "image_registrator",
            options={
                "database_path": database_path,
                "input_path": model_path,
                "output_path": output_path,
            },
        )
        self._progress(progress, "relocalization", 1, 1)
        return {
            "model_path": str(model_path),
            "output_path": str(output_path),
            "requested_image_ids": [int(i) for i in image_ids or []],
            "engine": "spheresfm image_registrator",
            "command": result,
        }

    def merge_reconstructions(
        self,
        *,
        model_paths: list[Path],
        output_path: Path,
        sim3_aligners: Any = None,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        """Merge N sparse models with ``model_merger`` (pairwise fold-left).

        Capability ``recon.merge``. ``model_merger`` is binary, so for
        N models it folds left: ``merge(merge(m0, m1), m2) ...``,
        threading the running result through a temp directory between
        steps. ``sim3_aligners`` is accepted for protocol parity but the
        COLMAP-fork ``model_merger`` performs its own alignment, so it is
        recorded as metadata only.
        """
        self._require_executable()
        models = [Path(path) for path in model_paths]
        if len(models) < 2:
            raise ValidationError("merge_reconstructions requires at least two model paths")
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        results: list[dict[str, Any]] = []
        total = len(models) - 1
        with tempfile.TemporaryDirectory(prefix="spheresfm-merge-") as tmp:
            tmp_root = Path(tmp)
            running = models[0]
            for index, next_model in enumerate(models[1:], start=1):
                is_last = index == total
                step_output = output_path if is_last else tmp_root / f"step_{index}"
                step_output.mkdir(parents=True, exist_ok=True)
                self._progress(progress, "merge", index - 1, total)
                results.append(
                    self._run_colmap(
                        "model_merger",
                        options={
                            "input_path1": running,
                            "input_path2": next_model,
                            "output_path": step_output,
                        },
                    )
                )
                self._progress(progress, "merge", index, total)
                running = step_output
        return {
            "model_paths": [str(path) for path in models],
            "output_path": str(output_path),
            "sim3_aligners": sim3_aligners,
            "engine": "spheresfm model_merger",
            "steps": results,
        }

    def export(
        self,
        *,
        model_path: Path,
        output_path: Path,
        format: str,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        """Export a sparse model with ``model_converter``.

        Capabilities ``export.ply`` / ``export.colmap_text`` /
        ``export.colmap_bin`` / ``export.nvm`` map to ``model_converter
        --output_type {PLY,TXT,BIN,NVM}``.
        """
        self._require_executable()
        normalized = str(format).replace("-", "_").lower()
        output_type = EXPORT_FORMAT_OUTPUT_TYPES.get(normalized)
        if output_type is None:
            raise CapabilityUnavailableError(
                capability="export.ply",
                reason=(
                    "SphereSfM exports ply, colmap_text, colmap_bin, and nvm; "
                    f"{format!r} is not supported."
                ),
            )
        model_path = Path(model_path)
        output_path = Path(output_path)
        # COLMAP writes TXT/BIN into a directory; PLY/NVM into a file.
        if output_type in ("TXT", "BIN"):
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        self._progress(progress, "export", 0, 1)
        result = self._run_colmap(
            "model_converter",
            options={
                "input_path": model_path,
                "output_path": output_path,
                "output_type": output_type,
            },
        )
        self._progress(progress, "export", 1, 1)
        return {
            "model_path": str(model_path),
            "output_path": str(output_path),
            "format": normalized,
            "output_type": output_type,
            "engine": "spheresfm model_converter",
            "command": result,
        }

    def apply_sim3(
        self,
        *,
        model_path: Path,
        output_path: Path,
        sim3: dict[str, Any],
        progress: Any | None = None,
    ) -> dict[str, Any]:
        """Apply a Sim(3) transform with ``model_transformer``/``model_aligner``.

        Capability ``georegister.sim3``. A caller-supplied transform
        (``sim3["transform_path"]`` — a COLMAP 3x4/4x4 TXT matrix) is
        applied via ``model_transformer``. When georeferenced inputs are
        supplied instead (``sim3["ref_images_path"]``) the transform is
        *solved and applied* via ``model_aligner``.
        """
        self._require_executable()
        model_path = Path(model_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        sim3 = dict(sim3 or {})

        transform_path = sim3.get("transform_path") or sim3.get("transform_txt_path")
        ref_images_path = sim3.get("ref_images_path") or sim3.get("ref_path")
        if transform_path:
            command = "model_transformer"
            options: dict[str, Any] = {
                "input_path": model_path,
                "output_path": output_path,
                "transform_path": transform_path,
            }
        elif ref_images_path:
            command = "model_aligner"
            options = {
                "input_path": model_path,
                "output_path": output_path,
                "ref_images_path": ref_images_path,
            }
        else:
            raise ValidationError(
                "apply_sim3 requires either a 'transform_path' (model_transformer) "
                "or a 'ref_images_path' (model_aligner) in the sim3 spec."
            )
        overrides = sim3.get("options") or {}
        if isinstance(overrides, dict):
            options.update(overrides)

        self._progress(progress, "apply_sim3", 0, 1)
        result = self._run_colmap(command, options=options)
        self._progress(progress, "apply_sim3", 1, 1)
        return {
            "model_path": str(model_path),
            "output_path": str(output_path),
            "engine": f"spheresfm {command}",
            "command": result,
        }

    def list_backend_config_schemas(self, *, include_schemas: bool = True) -> list[dict[str, Any]]:
        """Provider-specific option schemas for the portable stages above.

        Narrower than the action catalog: it only describes the option
        knobs that can ride through sfmapi's ``backend_options`` envelope
        for the feature, pair-selection, and mapping stages this backend
        implements as portable capabilities.
        """
        if self._find_executable() is None:
            return []
        rows: list[dict[str, Any]] = [
            {
                "config_id": "spheresfm.features.sift",
                "backend": self.name,
                "stage": "features",
                "capability": "features.extract.sift",
                "provider": "spheresfm",
                "display_name": "SphereSfM SPHERE feature extraction options",
                "description": (
                    "SphereSfM `feature_extractor` knobs for the SPHERE camera "
                    "model used by the portable features.extract.sift stage."
                ),
                "metadata": {"family": "spheresfm", "command": "feature_extractor"},
            },
            {
                "config_id": "spheresfm.pairs.exhaustive",
                "backend": self.name,
                "stage": "pairs",
                "capability": "pairs.exhaustive",
                "provider": "spheresfm",
                "display_name": "SphereSfM exhaustive matching options",
                "description": (
                    "SphereSfM `exhaustive_matcher` knobs for the portable "
                    "pairs.exhaustive stage (geometric verification runs inline)."
                ),
                "metadata": {"family": "spheresfm", "command": "exhaustive_matcher"},
            },
            {
                "config_id": "spheresfm.mapping.spherical",
                "backend": self.name,
                "stage": "mapping",
                "capability": "map.spherical",
                "provider": "spheresfm",
                "display_name": "SphereSfM spherical mapper options",
                "description": (
                    "SphereSfM `mapper` knobs for the portable map.spherical "
                    "stage; runs with --Mapper.sphere_camera 1."
                ),
                "metadata": {"family": "spheresfm", "command": "mapper"},
            },
        ]
        if include_schemas:
            for row in rows:
                row["option_schema"] = self._stage_option_schema(str(row["config_id"]))
        return rows

    def list_backend_artifact_contracts(self) -> list[dict[str, Any]]:
        """Native artifact formats the portable SphereSfM stages exchange.

        SphereSfM is a COLMAP fork: the features/pairs stages accumulate
        state in one SQLite database and the spherical mapper emits a
        COLMAP sparse model directory. These explicit contracts make the
        manifest's declared artifact-contract ids resolvable instead of
        leaving them as vaporware.
        """
        return [
            {
                "contract_id": "spheresfm.matches.database",
                "stage": "matcher",
                "capability": "pairs.exhaustive",
                "provider": "spheresfm",
                "display_name": "SphereSfM SQLite match database",
                "description": (
                    "COLMAP-format SQLite database (database.db) populated by "
                    "SphereSfM feature_extractor and the matchers. SphereSfM's "
                    "matchers run geometric verification inline, so the same "
                    "file carries keypoints, descriptors, raw matches, and "
                    "verified two-view geometries between the feature and "
                    "matcher stages."
                ),
                "accepts": ["matches.database.colmap"],
                "emits": ["matches.database.verified.colmap"],
                "preferred": "matches.database.verified.colmap",
            },
            {
                "contract_id": "spheresfm.reconstruction.spherical",
                "stage": "mapping",
                "capability": "map.spherical",
                "provider": "spheresfm",
                "display_name": "SphereSfM spherical sparse model",
                "description": (
                    "COLMAP sparse reconstruction directory (cameras/images/"
                    "points3D) emitted by the SphereSfM mapper with "
                    "--Mapper.sphere_camera 1, written as sparse/N sub-models "
                    "and consumed by sphere_cubic_reprojecer."
                ),
                "accepts": ["matches.database.verified.colmap"],
                "emits": ["reconstruction.sparse.colmap"],
                "preferred": "reconstruction.sparse.colmap",
            },
        ]

    def _stage_option_schema(self, config_id: str) -> dict[str, Any]:
        """Curated, runtime-managed-key-free option schema for a stage.

        SphereSfM's CLI does not emit a machine-readable option dump, so
        these schemas are a hand-picked safe subset. Runtime-managed keys
        (``database_path``, ``image_path``, ``output_path`` ...) are
        deliberately excluded — sfmapi supplies them.
        """
        if config_id == "spheresfm.features.sift":
            properties: dict[str, Any] = {
                "camera_params": {
                    "type": "string",
                    "description": "SPHERE camera params 'f,cx,cy' (default 1,3520,1760).",
                },
                "single_camera": {"type": "boolean"},
                "ImageReader.camera_mask_path": {"type": "string"},
                "ImageReader.pose_path": {"type": "string"},
                "SiftExtraction.max_num_features": {"type": "integer"},
                "SiftExtraction.use_gpu": {"type": "boolean"},
            }
        elif config_id == "spheresfm.pairs.exhaustive":
            properties = {
                "SiftMatching.max_error": {"type": "number"},
                "SiftMatching.min_num_inliers": {"type": "integer"},
                "SiftMatching.use_gpu": {"type": "boolean"},
                "ExhaustiveMatching.block_size": {"type": "integer"},
            }
        elif config_id == "spheresfm.mapping.spherical":
            properties = {
                "Mapper.sphere_camera": {"type": "integer"},
                "Mapper.ba_refine_focal_length": {"type": "integer"},
                "Mapper.ba_refine_principal_point": {"type": "integer"},
                "Mapper.ba_refine_extra_params": {"type": "integer"},
                "Mapper.min_num_matches": {"type": "integer"},
            }
        else:  # pragma: no cover - defensive
            properties = {}
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
        }

    def list_backend_actions(self, *, include_schemas: bool = False) -> list[dict[str, Any]]:
        actions = [
            self._reconstruct_action(include_schemas=include_schemas),
            self._cubemap_action(include_schemas=include_schemas),
        ]
        actions.extend(
            self._command_action(command, include_schemas=include_schemas)
            for command in SPHERESFM_COMMANDS
        )
        return sorted(actions, key=lambda action: str(action["action_id"]))

    def get_backend_action(self, action_id: str) -> dict[str, Any]:
        for action in self.list_backend_actions(include_schemas=True):
            if action["action_id"] == action_id:
                return action
        raise NotFoundError(f"Backend action {action_id!r} not found")

    def validate_backend_action(self, action_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            normalized = self._normalize_action_inputs(action_id, dict(inputs or {}))
        except ValidationError as exc:
            return {
                "action_id": action_id,
                "valid": False,
                "errors": [{"field": None, "message": str(exc)}],
                "normalized_inputs": {},
            }
        return {
            "action_id": action_id,
            "valid": True,
            "errors": [],
            "normalized_inputs": normalized,
        }

    def run_backend_action(
        self,
        action_id: str,
        inputs: dict[str, Any],
        *,
        workspace: Path | None = None,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        normalized = self._normalize_action_inputs(action_id, dict(inputs or {}))
        if action_id == "spheresfm.reconstructPanoramaFolder":
            return self._run_reconstruct(normalized, workspace=workspace, progress=progress)
        if action_id == "spheresfm.convertToCubemap":
            return self._run_convert_to_cubemap(normalized, progress=progress)
        if action_id.startswith("spheresfm.colmap."):
            command = action_id.removeprefix("spheresfm.colmap.")
            return self._run_generic_command(command, normalized)
        raise NotFoundError(f"Backend action {action_id!r} not found")

    def _find_executable(self) -> Path | None:
        if self._executable_override is not None:
            return self._executable_override if self._executable_override.exists() else None
        return resolve_spheresfm_executable(None)

    def _require_executable(self) -> Path:
        executable = self._find_executable()
        if executable is None:
            raise CapabilityUnavailableError(
                capability="backend.actions",
                reason=(
                    "SphereSfM executable not found. Build third_party/spheresfm and set "
                    "SFMAPI_SPHERESFM_EXECUTABLE."
                ),
            )
        return executable

    def _git_revision(self, root: Path) -> str | None:
        if not root.exists():
            return None
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except OSError:
            return None
        value = completed.stdout.strip()
        return value or None

    def _run_reconstruct(
        self,
        inputs: dict[str, Any],
        *,
        workspace: Path | None,
        progress: Any | None,
    ) -> dict[str, Any]:
        image_path = Path(str(inputs["image_path"]))
        workspace_path = Path(str(inputs["workspace_path"]))
        database_path = Path(str(inputs.get("database_path") or workspace_path / "database.db"))
        sparse_path = Path(str(inputs.get("sparse_path") or workspace_path / "sparse"))
        workspace_path.mkdir(parents=True, exist_ok=True)
        sparse_path.mkdir(parents=True, exist_ok=True)

        matching_mode = str(inputs.get("matching_mode", "spatial"))
        commands: list[tuple[str, dict[str, Any]]] = [
            ("database_creator", {"database_path": database_path}),
            (
                "feature_extractor",
                {
                    "database_path": database_path,
                    "image_path": image_path,
                    "ImageReader.camera_model": "SPHERE",
                    "ImageReader.camera_params": inputs.get("camera_params", "1,3520,1760"),
                    "ImageReader.single_camera": int(bool(inputs.get("single_camera", True))),
                },
            ),
            (
                self._matcher_command(matching_mode),
                self._matcher_options(matching_mode, inputs, database_path),
            ),
            (
                "mapper",
                {
                    "database_path": database_path,
                    "image_path": image_path,
                    "output_path": sparse_path,
                    "Mapper.ba_refine_focal_length": 0,
                    "Mapper.ba_refine_principal_point": 0,
                    "Mapper.ba_refine_extra_params": 0,
                    "Mapper.sphere_camera": 1,
                },
            ),
        ]
        if inputs.get("camera_mask_path"):
            commands[1][1]["ImageReader.camera_mask_path"] = inputs["camera_mask_path"]
        if inputs.get("pose_path"):
            commands[1][1]["ImageReader.pose_path"] = inputs["pose_path"]
        if inputs.get("use_gpu") is not None:
            use_gpu = int(bool(inputs["use_gpu"]))
            commands[1][1]["SiftExtraction.use_gpu"] = use_gpu
            commands[2][1]["SiftMatching.use_gpu"] = use_gpu
        if inputs.get("max_num_features") is not None:
            commands[1][1]["SiftExtraction.max_num_features"] = int(inputs["max_num_features"])

        results: list[dict[str, Any]] = []
        total = len(commands)
        for index, (command, options) in enumerate(commands, start=1):
            self._progress(progress, command, index - 1, total)
            results.append(
                self._run_colmap(
                    command,
                    options=options,
                    timeout_seconds=inputs.get("timeout_seconds"),
                )
            )
            self._progress(progress, command, index, total)
        return {
            "steps": results,
            "image_path": str(image_path),
            "workspace_path": str(workspace_path),
            "database_path": str(database_path),
            "sparse_path": str(sparse_path),
        }

    def _run_convert_to_cubemap(
        self,
        inputs: dict[str, Any],
        *,
        progress: Any | None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "image_path": inputs["image_path"],
            "input_path": inputs["input_path"],
            "output_path": inputs["output_path"],
        }
        for key in ("image_ids", "image_size", "field_of_view"):
            if key in inputs:
                options[key] = inputs[key]
        self._progress(progress, "sphere_cubic_reprojecer", 0, 1)
        result = self._run_colmap(
            "sphere_cubic_reprojecer",
            options=options,
            timeout_seconds=inputs.get("timeout_seconds"),
        )
        self._progress(progress, "sphere_cubic_reprojecer", 1, 1)
        return result

    def _run_generic_command(self, command: str, inputs: dict[str, Any]) -> dict[str, Any]:
        self._validate_command(command)
        options = dict(inputs.get("options") or {})
        positional = [str(arg) for arg in inputs.get("args", [])]
        return self._run_colmap(
            command,
            options=options,
            positional=positional,
            cwd=Path(str(inputs["cwd"])) if inputs.get("cwd") else None,
            timeout_seconds=inputs.get("timeout_seconds"),
        )

    def _run_colmap(
        self,
        command: str,
        *,
        options: dict[str, Any] | None = None,
        positional: list[str] | None = None,
        cwd: Path | None = None,
        timeout_seconds: int | float | None = None,
    ) -> dict[str, Any]:
        executable = self._require_executable()
        args = [str(executable), command, *(positional or [])]
        for key, value in (options or {}).items():
            if value is None:
                continue
            args.append(f"--{key}")
            args.append(self._stringify(value))
        try:
            completed = subprocess.run(
                args,
                check=True,
                capture_output=True,
                text=True,
                cwd=str(cwd) if cwd else None,
                timeout=timeout_seconds,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            raise ValidationError(f"SphereSfM command failed: {detail}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ValidationError(f"SphereSfM command timed out after {timeout_seconds}s") from exc
        return {
            "command": command,
            "args": args,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    def _matcher_command(self, mode: str) -> str:
        if mode == "vocabtree":
            return "vocab_tree_matcher"
        if mode == "exhaustive":
            return "exhaustive_matcher"
        if mode == "sequential":
            return "sequential_matcher"
        return "spatial_matcher"

    def _matcher_options(
        self,
        mode: str,
        inputs: dict[str, Any],
        database_path: Path,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {"database_path": database_path}
        if mode == "spatial":
            options.update(
                {
                    "SiftMatching.max_error": inputs.get("sift_max_error", 4),
                    "SiftMatching.min_num_inliers": inputs.get("sift_min_num_inliers", 50),
                    "SpatialMatching.is_gps": int(bool(inputs.get("spatial_is_gps", False))),
                    "SpatialMatching.max_distance": inputs.get("spatial_max_distance", 50),
                }
            )
        if mode == "vocabtree" and inputs.get("vocab_tree_path"):
            options["VocabTreeMatching.vocab_tree_path"] = inputs["vocab_tree_path"]
        return options

    def _reconstruct_action(self, *, include_schemas: bool) -> dict[str, Any]:
        descriptor = {
            "action_id": "spheresfm.reconstructPanoramaFolder",
            "backend": self.name,
            "display_name": "SphereSfM panorama reconstruction",
            "description": "Run the documented spherical-image feature, matching, and mapper sequence.",
            "category": "pipeline",
            "stability": "backend_extension",
            "side_effects": "write",
            "long_running": True,
            "supports_progress": True,
            "idempotent": False,
            "gpu_required": True,
            "required_capabilities": [],
            "metadata": {"family": "spheresfm", "source": "SphereSfM README command sequence"},
        }
        if include_schemas:
            descriptor["input_schema"] = self._reconstruct_input_schema()
            descriptor["output_schema"] = self._run_output_schema()
        return descriptor

    def _cubemap_action(self, *, include_schemas: bool) -> dict[str, Any]:
        descriptor = {
            "action_id": "spheresfm.convertToCubemap",
            "backend": self.name,
            "display_name": "SphereSfM cubic reprojection",
            "description": "Convert a spherical reconstruction to cubic/perspective images.",
            "category": "spherical",
            "stability": "backend_extension",
            "side_effects": "write",
            "long_running": True,
            "supports_progress": True,
            "idempotent": False,
            "gpu_required": False,
            "required_capabilities": [],
            "metadata": {"family": "spheresfm", "command": "sphere_cubic_reprojecer"},
        }
        if include_schemas:
            descriptor["input_schema"] = self._cubemap_input_schema()
            descriptor["output_schema"] = self._run_output_schema()
        return descriptor

    def _command_action(self, command: str, *, include_schemas: bool) -> dict[str, Any]:
        read_only = command in READ_ONLY_COMMANDS
        descriptor = {
            "action_id": f"spheresfm.colmap.{command}",
            "backend": self.name,
            "display_name": f"SphereSfM {command}",
            "description": f"Run the upstream SphereSfM `colmap {command}` command.",
            "category": self._command_category(command),
            "stability": "backend_extension",
            "side_effects": "read" if read_only else "write",
            "long_running": not read_only,
            "supports_progress": False,
            "idempotent": read_only,
            "gpu_required": command
            in {
                "feature_extractor",
                "exhaustive_matcher",
                "sequential_matcher",
                "spatial_matcher",
                "vocab_tree_matcher",
                "patch_match_stereo",
            },
            "required_capabilities": [],
            "metadata": {"family": "spheresfm", "command": command},
        }
        if include_schemas:
            descriptor["input_schema"] = self._generic_command_input_schema(command)
            descriptor["output_schema"] = self._run_output_schema()
        return descriptor

    def _reconstruct_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["image_path", "workspace_path"],
            "properties": {
                "image_path": {"type": "string"},
                "workspace_path": {"type": "string"},
                "database_path": {"type": "string"},
                "sparse_path": {"type": "string"},
                "camera_params": {"type": "string", "default": "1,3520,1760"},
                "single_camera": {"type": "boolean", "default": True},
                "use_gpu": {"type": "boolean", "default": True},
                "max_num_features": {"type": "integer", "default": 8192},
                "camera_mask_path": {"type": "string"},
                "pose_path": {"type": "string"},
                "matching_mode": {
                    "type": "string",
                    "enum": sorted(MATCHING_MODES),
                    "default": "spatial",
                },
                "vocab_tree_path": {"type": "string"},
                "sift_max_error": {"type": "number", "default": 4},
                "sift_min_num_inliers": {"type": "integer", "default": 50},
                "spatial_is_gps": {"type": "boolean", "default": False},
                "spatial_max_distance": {"type": "number", "default": 50},
                "timeout_seconds": {"type": "number"},
            },
        }

    def _cubemap_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["image_path", "input_path", "output_path"],
            "properties": {
                "image_path": {"type": "string"},
                "input_path": {"type": "string"},
                "output_path": {"type": "string"},
                "image_ids": {"type": "string", "default": "0,1,2,3,4,5"},
                "image_size": {"type": "integer", "default": 0},
                "field_of_view": {"type": "number", "default": 45.0},
                "timeout_seconds": {"type": "number"},
            },
        }

    def _generic_command_input_schema(self, command: str) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"Positional args passed after `colmap {command}`.",
                },
                "options": {
                    "type": "object",
                    "additionalProperties": {"type": ["string", "number", "integer", "boolean"]},
                    "description": "Named COLMAP/SphereSfM options without leading `--`.",
                },
                "cwd": {"type": "string"},
                "timeout_seconds": {"type": "number"},
            },
        }

    def _run_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "args": {"type": "array", "items": {"type": "string"}},
                "returncode": {"type": "integer"},
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
            },
        }

    def _normalize_action_inputs(self, action_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
        self.get_backend_action(action_id)
        if action_id == "spheresfm.reconstructPanoramaFolder":
            for field in ("image_path", "workspace_path"):
                if not inputs.get(field):
                    raise ValidationError(f"{field} is required")
            matching_mode = str(inputs.get("matching_mode", "spatial"))
            if matching_mode not in MATCHING_MODES:
                raise ValidationError(
                    f"matching_mode must be one of: {', '.join(sorted(MATCHING_MODES))}"
                )
            inputs["matching_mode"] = matching_mode
            return inputs
        if action_id == "spheresfm.convertToCubemap":
            for field in ("image_path", "input_path", "output_path"):
                if not inputs.get(field):
                    raise ValidationError(f"{field} is required")
            return inputs
        if action_id.startswith("spheresfm.colmap."):
            command = action_id.removeprefix("spheresfm.colmap.")
            self._validate_command(command)
            args = inputs.get("args", [])
            if args is None:
                args = []
            if not isinstance(args, list):
                raise ValidationError("args must be an array of strings")
            options = inputs.get("options", {})
            if options is None:
                options = {}
            if not isinstance(options, dict):
                raise ValidationError("options must be an object")
            inputs["args"] = [str(arg) for arg in args]
            inputs["options"] = dict(options)
            return inputs
        raise NotFoundError(f"Backend action {action_id!r} not found")

    def _validate_command(self, command: str) -> None:
        if command == "gui":
            raise ValidationError("SphereSfM GUI is not exposed through sfmapi")
        if command not in SPHERESFM_COMMAND_SET:
            raise ValidationError(f"unknown SphereSfM command: {command!r}")

    def _command_category(self, command: str) -> str:
        if "matcher" in command:
            return "matching"
        if command.startswith("feature_"):
            return "features"
        if "mapper" in command or command in {"bundle_adjuster", "point_triangulator"}:
            return "mapping"
        if command.startswith("model_") or command.startswith("image_"):
            return "model"
        if command in {"patch_match_stereo", "stereo_fusion", "poisson_mesher", "delaunay_mesher"}:
            return "dense"
        if command.startswith("database_"):
            return "database"
        if command.startswith("sphere_"):
            return "spherical"
        return "utility"

    def _progress(self, progress: Any | None, phase: str, current: int, total: int) -> None:
        if progress is None:
            return
        try:
            progress.phase_progress(f"spheresfm.{phase}", current=current, total=total)
        except Exception:
            return

    def _stringify(self, value: Any) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        return str(value)


__all__ = [
    "DEFAULT_SPHERESFM_ROOT",
    "DEFAULT_SPHERE_CAMERA_PARAMS",
    "EXPORT_FORMAT_OUTPUT_TYPES",
    "MATCHING_MODES",
    "PAIR_MODE_CAPABILITIES",
    "PAIR_MODE_COMMANDS",
    "SPHERESFM_CAPABILITIES",
    "SPHERESFM_COMMANDS",
    "SPHERESFM_COMMAND_SET",
    "SphereSfMBackend",
    "configure_spheresfm_environment",
    "resolve_spheresfm_executable",
]
