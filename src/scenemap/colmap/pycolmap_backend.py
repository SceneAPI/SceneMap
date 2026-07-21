from __future__ import annotations

import contextlib
import importlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from .model import read_colmap_text_model
from .pycolmap.backend import (
    COLMAP_EXPORT_TYPES,
    COLMAP_PAIR_ID_BASE,
    MATCH_COMMANDS,
    REPO_ROOT,
    CapabilityUnavailableError,
    ColmapCliBackend,
    ValidationError,
    colmap_runtime_path_dirs,
)

_DLL_DIRECTORY_HANDLES: list[Any] = []
_DLL_DIRECTORIES_REGISTERED = False


def register_pycolmap_dll_directories(executable: Path | None = None) -> None:
    """Register likely Windows DLL directories before importing PyCOLMAP."""

    global _DLL_DIRECTORIES_REGISTERED
    if os.name != "nt" or _DLL_DIRECTORIES_REGISTERED or not hasattr(os, "add_dll_directory"):
        return

    for resolved in colmap_runtime_path_dirs(executable):
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(resolved)))

    _DLL_DIRECTORIES_REGISTERED = True


class PycolmapBackend(ColmapCliBackend):
    """sfmapi backend that calls PyCOLMAP bindings before falling back to CLI-only tools."""

    name = "colmap_pycolmap"
    version = "0.0.1"
    vendor = "COLMAP upstream / PyCOLMAP"

    def capabilities(self) -> set[str]:
        try:
            pycolmap = self._require_pycolmap("capabilities")
        except CapabilityUnavailableError:
            return set()

        capabilities = {
            "features.extract.sift",
            "matches.verify",
            "pairs.exhaustive",
            "pairs.sequential",
            "pairs.spatial",
            "pairs.vocabtree",
            "pairs.from_poses",
            "matchers.nn-mutual",
            "matchers.nn-ratio",
            "map.incremental",
            "map.global",
            "ba.standard",
            "triangulate.retri",
            "export.ply",
            "export.colmap_text",
            "export.colmap_bin",
            # pycolmap runs in-process (no subprocess) but still
            # materializes a COLMAP database + sparse model on disk, so
            # ``compute.in_memory`` is intentionally NOT advertised — that
            # flag is for backends with zero on-disk artifact lifetime.
            "localize.from_memory",
            "geometry.two_view",
            # ``pycolmap.align_reconstruction_to_locations`` is a direct
            # binding; georegistration from GPS works without the CLI.
            "georegister.gps",
            # ``incremental_mapping`` consumes ``pycolmap.PosePrior`` rows.
            "pose_priors.mapping",
        }
        if bool(getattr(pycolmap, "has_cuda", False)):
            capabilities.update(
                {
                    "backend.actions",
                }
            )
        if self._find_colmap() is not None:
            capabilities.update(
                {
                    "pairs.explicit",
                    "map.hierarchical",
                    "relocalize.images",
                    "recon.merge",
                    "export.nvm",
                    "georegister.sim3",
                    # CLI-only tools wrapped by the new protocol surfaces.
                    "pgo.optimize",
                    "image.undistort",
                    "index.vocab_tree",
                    "rigs.configure",
                }
            )
        return capabilities

    def extract_features(
        self,
        *,
        database_path: Path,
        image_root: Path,
        image_list: list[str],
        options: dict,
        progress: Any | None = None,
    ) -> dict:
        pycolmap = self._require_pycolmap("features.extract.sift")
        database_path.parent.mkdir(parents=True, exist_ok=True)

        reader_options, extraction_options, camera_mode = self._feature_options(pycolmap, options)
        total = len(image_list)
        self._progress(progress, "feature_extraction", current=0, total=total)
        pycolmap.extract_features(
            database_path,
            image_root,
            image_names=image_list,
            camera_mode=camera_mode,
            reader_options=reader_options,
            extraction_options=extraction_options,
            device=self._device(pycolmap, options, extraction_options),
        )
        self._progress(progress, "feature_extraction", current=total, total=total)
        return {
            "num_images": len(image_list),
            "database_path": str(database_path),
            "engine": "pycolmap.extract_features",
        }

    def match(
        self,
        *,
        database_path: Path,
        mode: str,
        options: dict,
        progress: Any | None = None,
    ) -> dict:
        normalized_mode = mode.replace("-", "_").lower()
        pycolmap = self._require_pycolmap(self._match_capability(normalized_mode))
        command = MATCH_COMMANDS.get(normalized_mode)
        if command == "matches_importer":
            if self._find_colmap() is not None:
                return super().match(
                    database_path=database_path,
                    mode=mode,
                    options=options,
                    progress=progress,
                )
            raise CapabilityUnavailableError(
                capability="pairs.explicit",
                reason="PyCOLMAP does not expose matches_importer; install/build the COLMAP CLI",
            )
        if command == "transitive_matcher":
            if self._find_colmap() is not None:
                return super().match(
                    database_path=database_path,
                    mode=mode,
                    options=options,
                    progress=progress,
                )
            raise CapabilityUnavailableError(
                capability="backend.actions",
                reason="PyCOLMAP does not expose transitive_matcher; install/build the COLMAP CLI",
            )

        match_function = {
            "exhaustive_matcher": pycolmap.match_exhaustive,
            "sequential_matcher": pycolmap.match_sequential,
            "spatial_matcher": pycolmap.match_spatial,
            "vocab_tree_matcher": pycolmap.match_vocabtree,
        }.get(command or "")
        if match_function is None:
            raise CapabilityUnavailableError(capability="backend.actions")

        matching_options, pairing_options, verification_options = self._matching_options(
            pycolmap,
            command or "",
            options,
        )
        total = self._match_progress_total(database_path, normalized_mode, options)
        self._progress(progress, "matching", current=0, total=total)
        match_function(
            database_path,
            matching_options=matching_options,
            pairing_options=pairing_options,
            verification_options=verification_options,
            device=self._device(pycolmap, options, matching_options),
        )
        if total is not None:
            self._progress(progress, "matching", current=total, total=total)
        return {
            "database_path": str(database_path),
            "strategy": mode,
            "engine": f"pycolmap.{getattr(match_function, '__name__', command)}",
        }

    def verify_matches(
        self,
        *,
        database_path: Path,
        options: dict,
        progress: Any | None = None,
    ) -> dict:
        pycolmap = self._require_pycolmap("matches.verify")
        pairs = self._database_match_pairs(database_path)
        total = len(pairs)
        self._progress(progress, "geometric_verification", current=0, total=total)
        if pairs:
            verification_options = self._two_view_geometry_options(pycolmap, options)
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
                pairs_path = Path(tmp.name)
                for image_name1, image_name2 in pairs:
                    tmp.write(f"{image_name1} {image_name2}\n")
            try:
                pycolmap.verify_matches(database_path, pairs_path, options=verification_options)
            finally:
                pairs_path.unlink(missing_ok=True)
        self._progress(progress, "geometric_verification", current=total, total=total)
        return {
            "database_path": str(database_path),
            "num_verified_pairs": sum(
                1 for _ in self.iter_two_view_geometries(database_path=database_path)
            ),
            "engine": "pycolmap.verify_matches",
        }

    def run_mapping(
        self,
        *,
        kind: str,
        db_path: Path,
        image_root: Path,
        sparse_root: Path,
        job_dir: Path,
        spec: dict,
        pose_priors: dict | None = None,
        progress: Any | None = None,
    ) -> tuple[list[dict], list[Any]]:
        pycolmap = self._require_pycolmap(f"map.{kind}")
        normalized_kind = kind.replace("-", "_").lower()
        sparse_root.mkdir(parents=True, exist_ok=True)
        job_dir.mkdir(parents=True, exist_ok=True)
        phase = self._mapping_progress_phase(normalized_kind)
        total = self._database_image_count(db_path)
        self._progress(progress, phase, current=0, total=total)

        # SCOPE: pose priors are consumed only by *incremental* mapping.
        # pycolmap's ``global_mapping`` / ``hierarchical_mapping`` (and the
        # CLI ``global_mapper`` / ``hierarchical_mapper`` this delegates to)
        # take no pose-prior input, so priors passed to those kinds are
        # silently ignored — a COLMAP limitation, not a plugin gap. The
        # ``pose_priors.mapping`` capability is therefore incremental-only.
        if normalized_kind == "incremental":
            pipeline_options = self._incremental_pipeline_options(pycolmap, spec)
            applied_priors = self._apply_pose_priors(pycolmap, db_path, pose_priors)
            if applied_priors:
                # COLMAP reads priors from the database's pose_priors table;
                # the pipeline only consumes them when explicitly enabled.
                self._enable_prior_positions(pipeline_options)
            reconstructions = pycolmap.incremental_mapping(
                db_path,
                image_root,
                sparse_root,
                options=pipeline_options,
            )
            engine = (
                "pycolmap.incremental_mapping+pose_priors"
                if applied_priors
                else "pycolmap.incremental_mapping"
            )
        elif normalized_kind in {"global", "glomap"}:
            pipeline_options = self._global_pipeline_options(pycolmap, spec)
            reconstructions = pycolmap.global_mapping(
                db_path,
                image_root,
                sparse_root,
                options=pipeline_options,
            )
            engine = "pycolmap.global_mapping"
        elif normalized_kind == "hierarchical" and self._find_colmap() is not None:
            return super().run_mapping(
                kind=kind,
                db_path=db_path,
                image_root=image_root,
                sparse_root=sparse_root,
                job_dir=job_dir,
                spec=spec,
                pose_priors=pose_priors,
                progress=progress,
            )
        else:
            raise CapabilityUnavailableError(
                capability=f"map.{kind}",
                reason="PyCOLMAP exposes incremental and global mapping; hierarchical mapping needs CLI",
            )

        summaries: list[dict] = []
        ordered_reconstructions: list[Any] = []
        for idx, reconstruction in sorted(dict(reconstructions).items()):
            model_dir = sparse_root / str(idx)
            if not model_dir.exists():
                model_dir.mkdir(parents=True, exist_ok=True)
                reconstruction.write(model_dir)

            text_dir = job_dir / "pycolmap_text_models" / str(idx)
            text_dir.mkdir(parents=True, exist_ok=True)
            reconstruction.write_text(text_dir)
            safe_reconstruction = read_colmap_text_model(text_dir)
            ordered_reconstructions.append(safe_reconstruction)
            summaries.append(
                {
                    "idx": int(idx) if str(idx).isdigit() else idx,
                    "num_reg_images": safe_reconstruction.num_reg_images(),
                    "num_points3D": len(safe_reconstruction.points3D),
                    "model_path": str(model_dir),
                    "engine": engine,
                }
            )
        if total is not None:
            self._progress(progress, phase, current=total, total=total)
        return summaries, ordered_reconstructions

    def bundle_adjustment(self, *, model_path: Path, output_path: Path, spec: dict) -> dict:
        pycolmap = self._require_pycolmap("ba.standard")
        output_path.mkdir(parents=True, exist_ok=True)
        reconstruction = self._read_pycolmap_reconstruction(pycolmap, model_path)
        pycolmap.bundle_adjustment(
            reconstruction,
            options=self._bundle_adjustment_options(pycolmap, spec),
        )
        reconstruction.write(output_path)
        return {
            "model_path": str(output_path),
            "mode": spec.get("mode", "standard"),
            "num_reg_images": self._num_reg_images(reconstruction),
            "num_points3D": self._num_points3d(reconstruction),
            "engine": "pycolmap.bundle_adjustment",
        }

    def triangulate(
        self,
        *,
        model_path: Path,
        database_path: Path,
        image_root: Path,
        output_path: Path,
    ) -> dict:
        pycolmap = self._require_pycolmap("triangulate.retri")
        output_path.mkdir(parents=True, exist_ok=True)
        reconstruction = self._read_pycolmap_reconstruction(pycolmap, model_path)
        result = pycolmap.triangulate_points(
            reconstruction,
            database_path,
            image_root,
            output_path,
        )
        return {
            "model_path": str(output_path),
            "num_reg_images": self._num_reg_images(result),
            "num_points3D": self._num_points3d(result),
            "engine": "pycolmap.triangulate_points",
        }

    def export(self, *, model_path: Path, output_path: Path, format: str) -> dict:
        pycolmap = self._require_pycolmap(f"export.{format}")
        format_key = format.replace("-", "_").lower()
        output_type = COLMAP_EXPORT_TYPES.get(format_key)
        if output_type is None:
            raise CapabilityUnavailableError(capability=f"export.{format}")

        reconstruction = self._read_pycolmap_reconstruction(pycolmap, model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_type == "TXT":
            output_path.mkdir(parents=True, exist_ok=True)
            reconstruction.write_text(output_path)
        elif output_type == "BIN":
            output_path.mkdir(parents=True, exist_ok=True)
            reconstruction.write(output_path)
        elif output_type == "PLY":
            reconstruction.export_PLY(output_path)
        elif self._find_colmap() is not None:
            return super().export(model_path=model_path, output_path=output_path, format=format)
        else:
            raise CapabilityUnavailableError(
                capability=f"export.{format}",
                reason=f"PyCOLMAP does not export {format}; install/build the COLMAP CLI",
            )
        return {"format": format_key, "output_path": str(output_path)}

    def read_reconstruction(self, path: Path) -> Any:
        pycolmap = self._require_pycolmap("reconstruction.read")
        return self._read_pycolmap_reconstruction(pycolmap, path)

    # ------------------------------------------------------------------
    # Tier 2/3 protocol surfaces wired onto the in-process pycolmap API.
    # ------------------------------------------------------------------

    def localize_from_memory(self, *, sparse_dir: Path, query_image: Path, spec: dict) -> dict:
        """Localize a single query image against a sealed sparse model.

        Uses ``pycolmap.estimate_and_refine_absolute_pose`` over 2D-3D
        correspondences. The sparse model carries no descriptors, so the
        COLMAP feature database that built it is required to recover
        reference descriptors — probed next to ``sparse_dir`` or supplied
        via ``spec["database_path"]``.
        """
        pycolmap = self._require_pycolmap("localize.from_memory")
        sparse_dir = Path(sparse_dir)
        query_image = Path(query_image)
        model_dir = self._resolve_model_dir(sparse_dir)
        if model_dir is None:
            raise CapabilityUnavailableError(
                capability="localize.from_memory",
                reason=f"no COLMAP sparse model found under {sparse_dir}",
            )
        database_path = self._resolve_reference_database(sparse_dir, spec)
        if database_path is None:
            raise CapabilityUnavailableError(
                capability="localize.from_memory",
                reason=(
                    "localization needs the COLMAP feature database that built "
                    "the model (sparse models carry no descriptors). Place "
                    "database.db beside the sparse dir or pass spec.database_path."
                ),
            )

        reconstruction = self._read_pycolmap_reconstruction(pycolmap, model_dir)
        ref_points3d, ref_descriptors = self._reference_descriptor_bank(
            reconstruction, database_path
        )
        if not ref_points3d:
            raise CapabilityUnavailableError(
                capability="localize.from_memory",
                reason="reference model has no triangulated points with descriptors",
            )

        query_keypoints, query_descriptors = self._extract_query_features(
            pycolmap, query_image, spec
        )
        if query_descriptors.shape[0] == 0:
            raise ValidationError("localize_from_memory: no features in query image")

        matched_2d, matched_3d = self._match_query_to_reference(
            query_keypoints, query_descriptors, ref_points3d, ref_descriptors
        )
        min_correspondences = int(spec.get("min_num_correspondences", 4))
        if len(matched_3d) < min_correspondences:
            return {
                "success": False,
                "num_correspondences": len(matched_3d),
                "engine": "pycolmap.estimate_and_refine_absolute_pose",
            }

        camera = self._query_camera(pycolmap, reconstruction, query_keypoints, spec)
        estimation_options = self._absolute_pose_estimation_options(pycolmap, spec)
        refinement_options = self._absolute_pose_refinement_options(pycolmap, spec)
        answer = pycolmap.estimate_and_refine_absolute_pose(
            matched_2d,
            matched_3d,
            camera,
            estimation_options=estimation_options,
            refinement_options=refinement_options,
        )
        if not answer:
            return {
                "success": False,
                "num_correspondences": len(matched_3d),
                "engine": "pycolmap.estimate_and_refine_absolute_pose",
            }
        return self._serialize_localization(answer, len(matched_3d), camera)

    def estimate_two_view_geometry(self, *, database_path: Path, spec: dict) -> dict:
        """Estimate E/F/H + relative pose for image pairs in a feature DB.

        Wraps the direct ``pycolmap.estimate_*`` callables. Pairs come
        either from ``spec["pairs"]`` (explicit image-name pairs) or from
        every match row already present in the database.
        """
        pycolmap = self._require_pycolmap("geometry.two_view")
        database_path = Path(database_path)
        if not database_path.exists():
            raise ValidationError(f"estimate_two_view_geometry: missing database {database_path}")

        model = str(spec.get("model") or spec.get("estimate") or "two_view").lower()
        pairs = self._two_view_pairs(database_path, spec)
        keypoint_cache: dict[int, list[list[float]]] = {}
        results: list[dict] = []
        for image_id1, image_id2 in pairs:
            points1 = self._cached_keypoint_xy(database_path, image_id1, keypoint_cache)
            points2 = self._cached_keypoint_xy(database_path, image_id2, keypoint_cache)
            matches = self._database_pair_matches(database_path, image_id1, image_id2)
            paired1 = [points1[i] for i, j in matches if i < len(points1) and j < len(points2)]
            paired2 = [points2[j] for i, j in matches if i < len(points1) and j < len(points2)]
            if len(paired1) < 5:
                continue
            estimate = self._estimate_pair_geometry(pycolmap, model, paired1, paired2, spec)
            if estimate is None:
                continue
            estimate.update({"image_id1": image_id1, "image_id2": image_id2})
            results.append(estimate)
        return {
            "database_path": str(database_path),
            "model": model,
            "num_pairs": len(results),
            "pairs": results,
            "engine": f"pycolmap.estimate_{model}",
        }

    def align_reconstruction(self, *, model_path: Path, output_path: Path, spec: dict) -> dict:
        """Solve + apply a georegistration transform from GPS / geo-tags.

        Prefers ``pycolmap.align_reconstruction_to_locations``; falls back
        to the COLMAP ``model_aligner`` CLI when image locations are not
        supplied inline and a GPS-tagged image directory is given instead.
        """
        pycolmap = self._require_pycolmap("georegister.gps")
        model_path = Path(model_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        locations = self._geo_locations_from_spec(spec)
        if locations:
            image_names, coords = locations
            reconstruction = self._read_pycolmap_reconstruction(pycolmap, model_path)
            ransac_options = self._ransac_options(pycolmap, spec)
            min_common = int(spec.get("min_common_images", 3))
            sim3 = pycolmap.align_reconstruction_to_locations(
                reconstruction,
                image_names,
                coords,
                min_common,
                ransac_options,
            )
            if sim3 is None:
                raise ValidationError(
                    "align_reconstruction: pycolmap could not align model to locations"
                )
            reconstruction.transform(sim3)
            reconstruction.write(output_path)
            return {
                "model_path": str(output_path),
                "num_reg_images": self._num_reg_images(reconstruction),
                "num_points3D": self._num_points3d(reconstruction),
                "num_locations": len(image_names),
                "engine": "pycolmap.align_reconstruction_to_locations",
            }

        if self._find_colmap() is not None:
            return self._align_reconstruction_cli(model_path, output_path, spec)
        raise CapabilityUnavailableError(
            capability="georegister.gps",
            reason=(
                "align_reconstruction needs inline image locations "
                "(spec.image_names + spec.locations) or the COLMAP CLI for "
                "ref_images_path-based alignment"
            ),
        )

    def undistort_images(
        self, *, model_path: Path, image_root: Path, output_path: Path, spec: dict
    ) -> dict:
        """Undistort images + emit adjusted intrinsics into ``output_path``.

        PyCOLMAP only exposes single-``Bitmap`` undistortion; the batch
        ``image_undistorter`` workflow that rewrites a whole model is
        CLI-only, so this delegates to the COLMAP CLI.
        """
        if self._find_colmap() is None:
            raise CapabilityUnavailableError(
                capability="image.undistort",
                reason=(
                    "PyCOLMAP exposes only single-image undistort_image(); the "
                    "batch image_undistorter needs the COLMAP CLI"
                ),
            )
        exe = self._require_colmap("image.undistort")
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        options = spec.get("undistort") or spec.get("options") or spec
        args = [
            exe,
            "image_undistorter",
            "--image_path",
            str(image_root),
            "--input_path",
            str(model_path),
            "--output_path",
            str(output_path),
        ]
        output_type = str(options.get("output_type") or "COLMAP")
        args.extend(["--output_type", output_type])
        for key, value in sorted(self._scalar_options(options).items()):
            if key == "output_type":
                continue
            args.extend([f"--{key}", self._stringify_option(value)])
        self._run(args)
        return {
            "output_path": str(output_path),
            "output_type": output_type,
            "engine": "colmap image_undistorter",
        }

    def build_vocab_tree(self, *, database_path: Path, output_path: Path, spec: dict) -> dict:
        """Build a reusable vocabulary-tree retrieval index from a feature DB.

        PyCOLMAP has no vocab-tree training binding; this wraps the COLMAP
        ``vocab_tree_builder`` CLI.
        """
        if self._find_colmap() is None:
            raise CapabilityUnavailableError(
                capability="index.vocab_tree",
                reason="PyCOLMAP has no vocab-tree builder binding; install the COLMAP CLI",
            )
        exe = self._require_colmap("index.vocab_tree")
        database_path = Path(database_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        options = spec.get("vocab_tree") or spec.get("options") or spec
        args = [
            exe,
            "vocab_tree_builder",
            "--database_path",
            str(database_path),
            "--vocab_tree_path",
            str(output_path),
        ]
        for key, value in sorted(self._scalar_options(options).items()):
            args.extend([f"--{key}", self._stringify_option(value)])
        self._run(args)
        return {
            "database_path": str(database_path),
            "vocab_tree_path": str(output_path),
            "engine": "colmap vocab_tree_builder",
        }

    def configure_rig(self, *, database_path: Path, spec: dict) -> dict:
        """Declare or calibrate a multi-camera rig over a feature database.

        Wraps the COLMAP ``rig_configurator`` CLI (PyCOLMAP exposes
        ``RigConfig`` but no DB-applying entrypoint).
        """
        if self._find_colmap() is None:
            raise CapabilityUnavailableError(
                capability="rigs.configure",
                reason="PyCOLMAP has no rig_configurator binding; install the COLMAP CLI",
            )
        exe = self._require_colmap("rigs.configure")
        database_path = Path(database_path)
        options = spec.get("rig") or spec.get("options") or spec
        args = [exe, "rig_configurator", "--database_path", str(database_path)]

        rig_config_path = options.get("rig_config_path") or spec.get("rig_config_path")
        cleanup_path: Path | None = None
        rig_config = spec.get("rig_config")
        if not isinstance(rig_config, (list, dict)):
            rig_config = None
        if rig_config is not None and not rig_config_path:
            fd, tmp_name = tempfile.mkstemp(suffix=".rig.json")
            cleanup_path = Path(tmp_name)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(rig_config, handle)
            rig_config_path = str(cleanup_path)
        try:
            if rig_config_path:
                args.extend(["--rig_config_path", str(rig_config_path)])
            for key, value in sorted(self._scalar_options(options).items()):
                if key in {"rig_config_path"}:
                    continue
                args.extend([f"--{key}", self._stringify_option(value)])
            self._run(args)
        finally:
            if cleanup_path is not None:
                cleanup_path.unlink(missing_ok=True)
        return {
            "database_path": str(database_path),
            "engine": "colmap rig_configurator",
        }

    def pose_graph_optimize(self, *, model_path: Path, output_path: Path, spec: dict) -> dict:
        """Run pose-graph optimization via COLMAP ``rotation_averager``.

        PyCOLMAP has no rotation-averaging binding, so this wraps the
        COLMAP CLI. The optimized rotations are written into a fresh model
        directory at ``output_path``.
        """
        if self._find_colmap() is None:
            raise CapabilityUnavailableError(
                capability="pgo.optimize",
                reason="PyCOLMAP has no rotation_averager binding; install the COLMAP CLI",
            )
        exe = self._require_colmap("pgo.optimize")
        model_path = Path(model_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        options = spec.get("pgo") or spec.get("options") or spec
        args = [
            exe,
            "rotation_averager",
            "--input_path",
            str(model_path),
            "--output_path",
            str(output_path),
        ]
        for key, value in sorted(self._scalar_options(options).items()):
            args.extend([f"--{key}", self._stringify_option(value)])
        self._run(args)
        return {
            "model_path": str(output_path),
            "engine": "colmap rotation_averager",
        }

    def runtime_versions(self) -> dict[str, str]:
        versions = {
            "backend": self.version,
            "colmap_source_sha": self._colmap_source_sha(),
            "colmap_source_version": self._colmap_source_version(),
        }
        try:
            pycolmap = self._require_pycolmap("runtime")
        except CapabilityUnavailableError:
            versions["pycolmap"] = "missing"
        else:
            versions["pycolmap"] = str(getattr(pycolmap, "__version__", "unknown"))
            versions["pycolmap_has_cuda"] = str(bool(getattr(pycolmap, "has_cuda", False))).lower()

        cli_versions = super().runtime_versions()
        for key in ("colmap_executable", "colmap_help_header"):
            if key in cli_versions:
                versions[key] = cli_versions[key]
        return versions

    def _require_pycolmap(self, capability: str) -> Any:
        register_pycolmap_dll_directories(self._find_colmap())
        try:
            return importlib.import_module("pycolmap")
        except ModuleNotFoundError as exc:
            raise CapabilityUnavailableError(
                capability=capability,
                reason=(
                    "PyCOLMAP is not installed. Install a wheel with `uv pip install pycolmap`, "
                    "or build the COLMAP 4.1 submodule with `-BuildPycolmap`."
                ),
            ) from exc
        except (ImportError, RuntimeError) as exc:
            raise CapabilityUnavailableError(
                capability=capability,
                reason=(
                    "PyCOLMAP is installed but its native runtime dependencies could not load. "
                    "Set SFMAPI_COLMAP_DLL_DIRS to the COLMAP/vcpkg/CUDA/cuDSS DLL directories. "
                    f"Original error: {exc}"
                ),
            ) from exc

    def _feature_options(self, pycolmap: Any, options: dict) -> tuple[Any, Any, Any]:
        reader_values: dict[str, Any] = {}
        extraction_values: dict[str, Any] = {}
        camera_mode_name = "AUTO"

        for parts, value in self._option_items(options):
            group, key = self._group_and_key(parts)
            if key in {"single_camera", "camera_mode", "intrinsics_mode"}:
                if self._truthy(value) or str(value).lower() in {"single", "single_camera"}:
                    camera_mode_name = "SINGLE"
                elif str(value).lower() == "per_image":
                    camera_mode_name = "PER_IMAGE"
                elif str(value).lower() == "per_folder":
                    camera_mode_name = "PER_FOLDER"
                continue
            if key == "type" and str(value).lower() in {"sift", "featureextractortype.sift"}:
                continue
            if group in {"imagereader", "reader"} or key in _IMAGE_READER_KEYS:
                reader_values[key] = value
            elif group in {"siftextraction", "sift"} or key in _SIFT_EXTRACTION_KEYS:
                self._put_nested(extraction_values, ("sift", key), value)
            elif group in {"featureextraction", "extraction"} or key in _FEATURE_EXTRACTION_KEYS:
                extraction_values[key] = value

        self._apply_default_gpu(extraction_values, "use_gpu")
        reader_options = pycolmap.ImageReaderOptions()
        extraction_options = pycolmap.FeatureExtractionOptions()
        self._merge_options(reader_options, reader_values, "ImageReader")
        self._merge_options(extraction_options, extraction_values, "FeatureExtraction")
        return (
            reader_options,
            extraction_options,
            getattr(pycolmap.CameraMode, camera_mode_name),
        )

    def _matching_options(self, pycolmap: Any, command: str, options: dict) -> tuple[Any, Any, Any]:
        matching_values: dict[str, Any] = {}
        pairing_values: dict[str, Any] = {}
        verification_values: dict[str, Any] = {}

        pairing_class = {
            "exhaustive_matcher": pycolmap.ExhaustivePairingOptions,
            "sequential_matcher": pycolmap.SequentialPairingOptions,
            "spatial_matcher": pycolmap.SpatialPairingOptions,
            "vocab_tree_matcher": pycolmap.VocabTreePairingOptions,
        }[command]
        pairing_keys = _PAIRING_KEYS[command]
        pairing_group = command.replace("_matcher", "matching").replace("_", "")

        for parts, value in self._option_items(options):
            group, key = self._group_and_key(parts)
            if key == "type" and str(value).lower() in {"nn-mutual", "nn_ratio", "nn-ratio"}:
                continue
            if group in {"featurematching", "matching"} or key in _FEATURE_MATCHING_KEYS:
                matching_values[key] = value
            elif group in {"siftmatching", "sift"} or key in _SIFT_MATCHING_KEYS:
                self._put_nested(matching_values, ("sift", key), value)
            elif group == pairing_group or key in pairing_keys:
                pairing_values[key] = value
            else:
                self._route_two_view_option(verification_values, group, key, value)

        self._apply_default_gpu(matching_values, "use_gpu")
        matching_options = pycolmap.FeatureMatchingOptions()
        pairing_options = pairing_class()
        verification_options = pycolmap.TwoViewGeometryOptions()
        self._merge_options(matching_options, matching_values, "FeatureMatching")
        self._merge_options(pairing_options, pairing_values, command)
        self._merge_options(verification_options, verification_values, "TwoViewGeometry")
        return matching_options, pairing_options, verification_options

    def _two_view_geometry_options(self, pycolmap: Any, options: dict) -> Any:
        values: dict[str, Any] = {}
        for parts, value in self._option_items(options):
            group, key = self._group_and_key(parts)
            self._route_two_view_option(values, group, key, value)
        two_view_options = pycolmap.TwoViewGeometryOptions()
        self._merge_options(two_view_options, values, "TwoViewGeometry")
        return two_view_options

    def _incremental_pipeline_options(self, pycolmap: Any, spec: dict) -> Any:
        values = self._pipeline_values(spec, _INCREMENTAL_PIPELINE_KEYS, _INCREMENTAL_MAPPER_KEYS)
        self._apply_default_gpu(values, "ba_use_gpu")
        options = pycolmap.IncrementalPipelineOptions()
        self._merge_options(options, values, "IncrementalPipeline")
        return options

    def _global_pipeline_options(self, pycolmap: Any, spec: dict) -> Any:
        values = self._pipeline_values(spec, _GLOBAL_PIPELINE_KEYS, set())
        options = pycolmap.GlobalPipelineOptions()
        self._merge_options(options, values, "GlobalPipeline")
        return options

    def _bundle_adjustment_options(self, pycolmap: Any, spec: dict) -> Any:
        source = spec.get("bundle_adjustment") or spec.get("options") or spec
        values: dict[str, Any] = {}
        for parts, value in self._option_items(source):
            group, key = self._group_and_key(parts)
            if key == "use_gpu":
                self._put_nested(values, ("ceres", "use_gpu"), value)
            elif group in {"ceresbundleadjustment", "ceres"} or key in _CERES_BA_KEYS:
                self._put_nested(values, ("ceres", key), value)
            elif group in {"bundleadjustment", "ba"} or key in _BUNDLE_ADJUSTMENT_KEYS:
                values[key] = value
        if "ceres" not in values:
            self._apply_default_gpu(values, ("ceres", "use_gpu"))
        options = pycolmap.BundleAdjustmentOptions()
        self._merge_options(options, values, "BundleAdjustment")
        return options

    def _pipeline_values(
        self,
        spec: dict,
        top_keys: set[str],
        nested_mapper_keys: set[str],
    ) -> dict[str, Any]:
        values: dict[str, Any] = {}
        source = spec.get("mapper") if isinstance(spec.get("mapper"), dict) else spec
        for parts, value in self._option_items(source):
            group, key = self._group_and_key(parts)
            if group in {"mapper", "incrementalmapper"} and key not in top_keys:
                self._put_nested(values, ("mapper", key), value)
            elif key in top_keys:
                values[key] = value
            elif key in nested_mapper_keys:
                self._put_nested(values, ("mapper", key), value)
        return values

    def _database_match_pairs(self, database_path: Path) -> list[tuple[str, str]]:
        with sqlite3.connect(database_path) as conn:
            image_names = {
                int(image_id): str(name)
                for image_id, name in conn.execute("select image_id, name from images").fetchall()
            }
            rows = conn.execute("select pair_id, rows from matches where rows > 0").fetchall()
        pairs: list[tuple[str, str]] = []
        for pair_id, _ in rows:
            image_id1, image_id2 = self._pair_id_to_image_ids(int(pair_id))
            if image_id1 in image_names and image_id2 in image_names:
                pairs.append((image_names[image_id1], image_names[image_id2]))
        return pairs

    def _read_pycolmap_reconstruction(self, pycolmap: Any, path: Path) -> Any:
        reconstruction = pycolmap.Reconstruction()
        reconstruction.read(path)
        return reconstruction

    # ---- pose-prior wiring for run_mapping ---------------------------

    def _apply_pose_priors(self, pycolmap: Any, db_path: Path, pose_priors: dict | None) -> int:
        """Write pose priors into the COLMAP database's pose_priors table.

        Returns the number of priors written. ``pose_priors`` maps an
        image name (or image id) to a position + optional covariance.
        COLMAP's incremental mapper reads priors from the database, so
        this is the canonical wiring; the actual ``incremental_mapping``
        call needs no extra argument.
        """
        if not pose_priors:
            return 0
        priors = pose_priors.get("priors") if isinstance(pose_priors, dict) else None
        if priors is None and isinstance(pose_priors, dict):
            priors = pose_priors
        if not isinstance(priors, dict) or not priors:
            return 0

        name_to_id = self._database_image_name_to_id(db_path)
        prior_class = getattr(pycolmap, "PosePrior", None)
        database = pycolmap.Database()
        database.open(str(db_path))
        written = 0
        try:
            for raw_key, payload in priors.items():
                image_id = self._resolve_prior_image_id(raw_key, name_to_id)
                if image_id is None:
                    continue
                prior = self._build_pose_prior(pycolmap, prior_class, payload)
                if prior is None:
                    continue
                try:
                    if database.exists_pose_prior(image_id):
                        database.update_pose_prior(image_id, prior)
                    else:
                        database.write_pose_prior(image_id, prior)
                except Exception:  # pragma: no cover - pycolmap build variance
                    continue
                written += 1
        finally:
            database.close()
        return written

    def _resolve_prior_image_id(self, raw_key: Any, name_to_id: dict[str, int]) -> int | None:
        if isinstance(raw_key, int):
            return raw_key
        text = str(raw_key)
        if text.isdigit():
            return int(text)
        return name_to_id.get(text)

    def _build_pose_prior(self, pycolmap: Any, prior_class: Any, payload: Any) -> Any:
        if prior_class is None:
            return None
        if isinstance(payload, dict):
            position = payload.get("position") or payload.get("xyz") or payload.get("translation")
        else:
            position = payload
        if position is None:
            return None
        try:
            coords = [float(v) for v in position]
        except (TypeError, ValueError):
            return None
        if len(coords) != 3:
            return None
        prior = prior_class()
        prior.position = coords
        if isinstance(payload, dict):
            covariance = payload.get("position_covariance") or payload.get("covariance")
            if covariance is not None:
                with contextlib.suppress(Exception):  # pragma: no cover - optional field
                    prior.position_covariance = covariance
            system = payload.get("coordinate_system")
            if system is not None:
                enum = getattr(pycolmap, "PosePriorCoordinateSystem", None)
                member = getattr(enum, str(system).upper(), None) if enum else None
                if member is not None:
                    prior.coordinate_system = member
        return prior

    def _enable_prior_positions(self, pipeline_options: Any) -> None:
        for attr in ("use_prior_position", "use_prior_motion"):
            if hasattr(pipeline_options, attr):
                try:
                    setattr(pipeline_options, attr, True)
                except Exception:  # pragma: no cover - pycolmap build variance
                    continue
        mapper = getattr(pipeline_options, "mapper", None)
        if mapper is not None and hasattr(mapper, "use_prior_position"):
            with contextlib.suppress(Exception):  # pragma: no cover
                mapper.use_prior_position = True

    def _database_image_name_to_id(self, database_path: Path) -> dict[str, int]:
        try:
            with sqlite3.connect(database_path) as conn:
                rows = conn.execute("select image_id, name from images").fetchall()
        except sqlite3.Error:
            return {}
        return {str(name): int(image_id) for image_id, name in rows}

    # ---- localize_from_memory helpers --------------------------------

    def _resolve_model_dir(self, sparse_dir: Path) -> Path | None:
        model_files = ("cameras.bin", "cameras.txt")
        if any((sparse_dir / name).exists() for name in model_files):
            return sparse_dir
        if sparse_dir.is_dir():
            for child in sorted(sparse_dir.iterdir()):
                if child.is_dir() and any((child / name).exists() for name in model_files):
                    return child
        return None

    def _resolve_reference_database(self, sparse_dir: Path, spec: dict) -> Path | None:
        explicit = spec.get("database_path") or spec.get("reference_database")
        if explicit:
            candidate = Path(explicit)
            return candidate if candidate.exists() else None
        candidates = [
            sparse_dir / "database.db",
            sparse_dir.parent / "database.db",
            sparse_dir.parent.parent / "database.db",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _reference_descriptor_bank(
        self, reconstruction: Any, database_path: Path
    ) -> tuple[list[list[float]], Any]:
        """Pool every triangulated 2D observation's descriptor.

        Returns ``(points3D_xyz, descriptors)`` aligned row-for-row so a
        query descriptor matched to row ``i`` yields ``points3D_xyz[i]``.
        """
        import numpy as np

        points3d: list[list[float]] = []
        descriptor_rows: list[Any] = []
        for image in reconstruction.images.values():
            descriptors, _dim = self._read_database_descriptors(database_path, image.image_id)
            if descriptors.shape[0] == 0:
                continue
            points2d = list(getattr(image, "points2D", []))
            for idx, point2d in enumerate(points2d):
                if idx >= descriptors.shape[0]:
                    break
                point3d_id = self._point2d_point3d_id(point2d)
                if point3d_id is None or point3d_id not in reconstruction.points3D:
                    continue
                xyz = reconstruction.points3D[point3d_id].xyz
                points3d.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
                descriptor_rows.append(descriptors[idx])
        if not descriptor_rows:
            return [], np.zeros((0, 128), dtype=np.float32)
        return points3d, np.vstack(descriptor_rows).astype(np.float32)

    def _point2d_point3d_id(self, point2d: Any) -> int | None:
        has_point3d = getattr(point2d, "has_point3D", None)
        if callable(has_point3d):
            if not has_point3d():
                return None
        elif has_point3d is False:
            return None
        point3d_id = getattr(point2d, "point3D_id", None)
        if point3d_id is None:
            return None
        point3d_id = int(point3d_id)
        invalid = getattr(point2d, "INVALID_POINT3D_ID", None)
        if invalid is not None and point3d_id == int(invalid):
            return None
        # COLMAP sentinel for "no 3D point" is the max uint64.
        if point3d_id >= 18446744073709551615:
            return None
        return point3d_id

    def _read_database_descriptors(self, database_path: Path, image_id: int) -> tuple[Any, int]:
        import numpy as np

        with sqlite3.connect(database_path) as conn:
            row = conn.execute(
                "select rows, cols, data from descriptors where image_id = ?",
                (int(image_id),),
            ).fetchone()
        if row is None:
            return np.zeros((0, 128), dtype=np.float32), 128
        rows, cols, blob = int(row[0]), int(row[1]), bytes(row[2] or b"")
        if rows <= 0 or cols <= 0 or len(blob) < rows * cols:
            return np.zeros((0, max(cols, 1)), dtype=np.float32), max(cols, 1)
        descriptors = np.frombuffer(blob[: rows * cols], dtype=np.uint8)
        return descriptors.reshape(rows, cols).astype(np.float32), cols

    def _extract_query_features(
        self, pycolmap: Any, query_image: Path, spec: dict
    ) -> tuple[Any, Any]:
        import numpy as np

        image_array = self._load_grayscale_image(query_image)
        # ``pycolmap.Sift`` takes a FeatureExtractionOptions whose nested
        # ``.sift`` member carries the SIFT detector knobs.
        extraction_options = pycolmap.FeatureExtractionOptions()
        sift_values = spec.get("sift") if isinstance(spec.get("sift"), dict) else {}
        if sift_values:
            try:
                extraction_options.sift.mergedict(dict(sift_values))
            except Exception as exc:
                raise ValidationError(f"invalid SiftExtraction options: {exc}") from exc
        # SIFT GPU extraction needs an attached GL/CUDA context; the CPU
        # path is self-contained and correct for single-image localization.
        if hasattr(extraction_options.sift, "use_gpu"):
            extraction_options.sift.use_gpu = False
        try:
            sift = pycolmap.Sift(options=extraction_options, device=pycolmap.Device.cpu)
        except TypeError:  # pragma: no cover - pycolmap build variance
            sift = pycolmap.Sift(extraction_options)
        keypoints, descriptors = sift.extract(image_array)
        return np.asarray(keypoints, dtype=np.float64), np.asarray(descriptors, dtype=np.float32)

    def _load_grayscale_image(self, query_image: Path) -> Any:
        import numpy as np

        try:
            from PIL import Image as PILImage
        except ModuleNotFoundError as exc:  # pragma: no cover - Pillow ships with sfmapi
            raise CapabilityUnavailableError(
                capability="localize.from_memory",
                reason="Pillow is required to decode the query image",
            ) from exc
        with PILImage.open(query_image) as handle:
            grayscale = handle.convert("L")
            return np.asarray(grayscale, dtype=np.uint8)

    def _match_query_to_reference(
        self,
        query_keypoints: Any,
        query_descriptors: Any,
        ref_points3d: list[list[float]],
        ref_descriptors: Any,
    ) -> tuple[Any, Any]:
        """Mutual nearest-neighbour match query → reference descriptors."""
        import numpy as np

        # COLMAP SIFT descriptors are L1-root normalised uint8; cast and
        # L2-normalise so a dot product is a cosine similarity.
        query = self._l2_normalize(query_descriptors)
        reference = self._l2_normalize(ref_descriptors)
        similarity = query @ reference.T
        query_best = np.argmax(similarity, axis=1)
        ref_best = np.argmax(similarity, axis=0)

        matched_2d: list[list[float]] = []
        matched_3d: list[list[float]] = []
        for query_idx, ref_idx in enumerate(query_best):
            if ref_best[ref_idx] != query_idx:
                continue
            matched_2d.append(
                [float(query_keypoints[query_idx][0]), float(query_keypoints[query_idx][1])]
            )
            matched_3d.append(ref_points3d[int(ref_idx)])
        return (
            np.asarray(matched_2d, dtype=np.float64).reshape(-1, 2),
            np.asarray(matched_3d, dtype=np.float64).reshape(-1, 3),
        )

    def _l2_normalize(self, descriptors: Any) -> Any:
        import numpy as np

        descriptors = np.asarray(descriptors, dtype=np.float32)
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return descriptors / norms

    def _query_camera(
        self, pycolmap: Any, reconstruction: Any, query_keypoints: Any, spec: dict
    ) -> Any:
        camera_spec = spec.get("camera") if isinstance(spec.get("camera"), dict) else None
        if camera_spec:
            return pycolmap.Camera(camera_spec)
        # Reuse the reference model's camera when the query was captured
        # by the same rig; otherwise synthesise a SIMPLE_RADIAL guess.
        for image in reconstruction.images.values():
            if image.camera is not None:
                return image.camera
        width = int(spec.get("width") or self._keypoint_extent(query_keypoints, 0))
        height = int(spec.get("height") or self._keypoint_extent(query_keypoints, 1))
        focal = float(spec.get("focal_length") or 1.2 * max(width, height, 1))
        return pycolmap.Camera.create_from_model_name(
            0, "SIMPLE_RADIAL", focal, max(width, 1), max(height, 1)
        )

    def _keypoint_extent(self, query_keypoints: Any, axis: int) -> int:
        if query_keypoints.shape[0] == 0:
            return 1
        return int(query_keypoints[:, axis].max()) + 1

    def _absolute_pose_estimation_options(self, pycolmap: Any, spec: dict) -> Any:
        options = pycolmap.AbsolutePoseEstimationOptions()
        values = spec.get("estimation") if isinstance(spec.get("estimation"), dict) else {}
        if values:
            try:
                options.mergedict(dict(values))
            except Exception as exc:
                raise ValidationError(f"invalid AbsolutePoseEstimation options: {exc}") from exc
        return options

    def _absolute_pose_refinement_options(self, pycolmap: Any, spec: dict) -> Any:
        options = pycolmap.AbsolutePoseRefinementOptions()
        values = spec.get("refinement") if isinstance(spec.get("refinement"), dict) else {}
        if values:
            try:
                options.mergedict(dict(values))
            except Exception as exc:
                raise ValidationError(f"invalid AbsolutePoseRefinement options: {exc}") from exc
        return options

    def _serialize_localization(self, answer: dict, num_correspondences: int, camera: Any) -> dict:
        cam_from_world = answer.get("cam_from_world")
        result: dict[str, Any] = {
            "success": True,
            "num_correspondences": num_correspondences,
            "engine": "pycolmap.estimate_and_refine_absolute_pose",
        }
        if cam_from_world is not None:
            rotation = getattr(cam_from_world, "rotation", None)
            translation = getattr(cam_from_world, "translation", None)
            quat = getattr(rotation, "quat", None) if rotation is not None else None
            if quat is not None:
                result["rotation"] = {
                    "x": float(quat[0]),
                    "y": float(quat[1]),
                    "z": float(quat[2]),
                    "w": float(quat[3]),
                }
            if translation is not None:
                result["translation"] = [float(v) for v in translation]
        inliers = answer.get("inliers")
        if inliers is not None:
            result["num_inliers"] = int(sum(1 for value in inliers if value))
        result["camera"] = {
            "model": str(getattr(camera, "model", "")),
            "width": int(getattr(camera, "width", 0)),
            "height": int(getattr(camera, "height", 0)),
            "params": [float(value) for value in getattr(camera, "params", [])],
        }
        return result

    # ---- estimate_two_view_geometry helpers --------------------------

    def _two_view_pairs(self, database_path: Path, spec: dict) -> list[tuple[int, int]]:
        explicit = spec.get("pairs") if isinstance(spec.get("pairs"), list) else None
        if explicit:
            name_to_id = self._database_image_name_to_id(database_path)
            pairs: list[tuple[int, int]] = []
            for pair in explicit:
                if isinstance(pair, dict):
                    first = pair.get("image_name1") or pair.get("image_id1")
                    second = pair.get("image_name2") or pair.get("image_id2")
                elif isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    first, second = pair[0], pair[1]
                else:
                    continue
                id1 = self._resolve_prior_image_id(first, name_to_id)
                id2 = self._resolve_prior_image_id(second, name_to_id)
                if id1 is not None and id2 is not None and id1 != id2:
                    pairs.append((id1, id2))
            return pairs
        return [
            (image_id1, image_id2)
            for image_id1, image_id2 in self._database_match_pair_ids(database_path)
        ]

    def _database_match_pair_ids(self, database_path: Path) -> list[tuple[int, int]]:
        try:
            with sqlite3.connect(database_path) as conn:
                rows = conn.execute("select pair_id from matches where rows > 0").fetchall()
        except sqlite3.Error:
            return []
        return [self._pair_id_to_image_ids(int(pair_id)) for (pair_id,) in rows]

    def _database_pair_matches(
        self, database_path: Path, image_id1: int, image_id2: int
    ) -> list[tuple[int, int]]:
        # COLMAP keys match rows by a pair_id built from the *ordered*
        # (smaller, larger) image ids, with match columns relative to that
        # order. Query with the canonical order (reusing the shared base
        # constant) and swap the returned columns back when the caller's
        # order was reversed — otherwise the row is not found and matches
        # are silently dropped, or returned mis-indexed.
        lo, hi = (image_id1, image_id2) if image_id1 <= image_id2 else (image_id2, image_id1)
        pair_id = lo * COLMAP_PAIR_ID_BASE + hi
        with sqlite3.connect(database_path) as conn:
            row = conn.execute(
                "select rows, cols, data from matches where pair_id = ?",
                (pair_id,),
            ).fetchone()
        if row is None:
            return []
        matches = self._decode_uint_matrix(int(row[0] or 0), int(row[1] or 0), row[2])
        if image_id1 > image_id2:
            matches = [(j, i) for (i, j) in matches]
        return matches

    def _cached_keypoint_xy(
        self,
        database_path: Path,
        image_id: int,
        cache: dict[int, list[list[float]]],
    ) -> list[list[float]]:
        if image_id not in cache:
            keypoints, _descriptors, _dim = self.read_keypoints(
                database_path=database_path, image_id=image_id
            )
            cache[image_id] = [[float(kp[0]), float(kp[1])] for kp in keypoints]
        return cache[image_id]

    def _estimate_pair_geometry(
        self,
        pycolmap: Any,
        model: str,
        points1: list[list[float]],
        points2: list[list[float]],
        spec: dict,
    ) -> dict | None:
        import numpy as np

        array1 = np.asarray(points1, dtype=np.float64).reshape(-1, 2)
        array2 = np.asarray(points2, dtype=np.float64).reshape(-1, 2)
        ransac = self._ransac_options(pycolmap, spec)
        if model in {"fundamental", "f"}:
            answer = pycolmap.estimate_fundamental_matrix(array1, array2, estimation_options=ransac)
            matrix_key = "fundamental_matrix"
        elif model in {"homography", "h"}:
            answer = pycolmap.estimate_homography_matrix(array1, array2, estimation_options=ransac)
            matrix_key = "homography_matrix"
        elif model in {"essential", "e"}:
            cameras = self._pair_cameras(pycolmap, spec)
            if cameras is None:
                raise ValidationError(
                    "estimate_two_view_geometry: essential model needs spec.camera1/camera2"
                )
            answer = pycolmap.estimate_essential_matrix(
                array1, array2, cameras[0], cameras[1], estimation_options=ransac
            )
            matrix_key = "essential_matrix"
        else:
            answer = self._estimate_two_view_default(pycolmap, array1, array2, ransac)
            matrix_key = None
        if not answer:
            return None
        result: dict[str, Any] = {"num_matches": int(array1.shape[0])}
        if matrix_key is not None and isinstance(answer, dict) and matrix_key in answer:
            result[matrix_key] = np.asarray(answer[matrix_key]).tolist()
        if isinstance(answer, dict):
            inliers = answer.get("inliers") or answer.get("inlier_mask")
            if inliers is not None:
                result["num_inliers"] = int(sum(1 for value in inliers if value))
            for key in ("cam2_from_cam1", "configuration_type", "config"):
                if key in answer:
                    result[key] = self._jsonify(answer[key])
        return result

    def _estimate_two_view_default(
        self, pycolmap: Any, array1: Any, array2: Any, ransac: Any
    ) -> Any:
        """Best-effort uncalibrated two-view estimate.

        ``estimate_two_view_geometry`` signatures vary across pycolmap
        builds; fall back to the well-defined fundamental-matrix
        estimator when the direct callable does not accept bare points.
        """
        estimator = getattr(pycolmap, "estimate_two_view_geometry", None)
        if estimator is not None:
            for attempt in (
                lambda: estimator(array1, array2),
                lambda: estimator(array1, array2, estimation_options=ransac),
            ):
                try:
                    return attempt()
                except TypeError:
                    continue
        return pycolmap.estimate_fundamental_matrix(array1, array2, estimation_options=ransac)

    def _pair_cameras(self, pycolmap: Any, spec: dict) -> tuple[Any, Any] | None:
        camera1 = spec.get("camera1")
        camera2 = spec.get("camera2") or camera1
        if not isinstance(camera1, dict) or not isinstance(camera2, dict):
            return None
        return pycolmap.Camera(camera1), pycolmap.Camera(camera2)

    def _jsonify(self, value: Any) -> Any:
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        return str(value)

    # ---- align_reconstruction helpers --------------------------------

    def _geo_locations_from_spec(self, spec: dict) -> tuple[list[str], Any] | None:
        import numpy as np

        image_names = spec.get("image_names") or spec.get("ref_image_names")
        locations = spec.get("locations") or spec.get("ref_locations")
        if image_names and locations is not None:
            names = [str(name) for name in image_names]
            coords = np.asarray(locations, dtype=np.float64).reshape(len(names), 3)
            return names, coords
        # Inline mapping form: {"image.jpg": [x, y, z], ...}
        mapping = spec.get("image_locations")
        if isinstance(mapping, dict) and mapping:
            names = [str(name) for name in mapping]
            coords = np.asarray([mapping[name] for name in names], dtype=np.float64).reshape(
                len(names), 3
            )
            return names, coords
        return None

    def _ransac_options(self, pycolmap: Any, spec: dict) -> Any:
        options = pycolmap.RANSACOptions()
        values = spec.get("ransac") if isinstance(spec.get("ransac"), dict) else {}
        if values:
            try:
                options.mergedict(dict(values))
            except Exception as exc:
                raise ValidationError(f"invalid RANSAC options: {exc}") from exc
        return options

    def _align_reconstruction_cli(self, model_path: Path, output_path: Path, spec: dict) -> dict:
        exe = self._require_colmap("georegister.gps")
        options = spec.get("aligner") or spec.get("options") or spec
        args = [
            exe,
            "model_aligner",
            "--input_path",
            str(model_path),
            "--output_path",
            str(output_path),
        ]
        ref_images_path = (
            spec.get("ref_images_path") or options.get("ref_images_path")
            if isinstance(options, dict)
            else spec.get("ref_images_path")
        )
        if ref_images_path:
            args.extend(["--ref_images_path", str(ref_images_path)])
        if isinstance(options, dict):
            for key, value in sorted(self._scalar_options(options).items()):
                if key in {"ref_images_path"}:
                    continue
                args.extend([f"--{key}", self._stringify_option(value)])
        self._run(args)
        rec = self.read_reconstruction(output_path)
        return {
            "model_path": str(output_path),
            "num_reg_images": rec.num_reg_images(),
            "num_points3D": len(rec.points3D),
            "engine": "colmap model_aligner",
        }

    # ---- shared CLI-option helpers -----------------------------------

    def _scalar_options(self, options: Any) -> dict[str, Any]:
        """Flatten a spec mapping to scalar CLI options.

        Drops nested containers and sfmapi routing keys so the result is
        safe to splat onto a COLMAP command line.
        """
        if not isinstance(options, dict):
            return {}
        skip = {
            "version",
            "type",
            "provider",
            "kind",
            "mode",
            "backend_options",
            "portable",
            "legacy_options",
            "options",
        }
        out: dict[str, Any] = {}
        for key, value in options.items():
            option_key = str(key)
            if option_key in skip or value is None:
                continue
            if isinstance(value, (dict, list, tuple)):
                continue
            out[option_key] = value
        return out

    def _device(self, pycolmap: Any, source_options: dict, option_obj: Any) -> Any:
        device_name = str(
            source_options.get("device") or os.environ.get("SFMAPI_COLMAP_DEVICE", "auto")
        )
        if device_name.lower() in {"cpu", "cuda", "auto"}:
            return getattr(pycolmap.Device, device_name.lower())
        use_gpu = getattr(option_obj, "use_gpu", None)
        if use_gpu is True:
            return pycolmap.Device.cuda
        if use_gpu is False:
            return pycolmap.Device.cpu
        return pycolmap.Device.auto

    def _apply_default_gpu(self, values: dict[str, Any], key: str | tuple[str, str]) -> None:
        raw = os.environ.get("SFMAPI_COLMAP_USE_GPU")
        if raw is None:
            return
        value = self._truthy(raw)
        if isinstance(key, tuple):
            current = values
            for part in key[:-1]:
                current = current.setdefault(part, {})
            current.setdefault(key[-1], value)
        else:
            values.setdefault(key, value)

    def _merge_options(self, option_obj: Any, values: dict[str, Any], label: str) -> None:
        if not values:
            return
        try:
            option_obj.mergedict(values)
        except Exception as exc:  # pragma: no cover - message depends on pycolmap internals
            raise ValidationError(f"invalid {label} options: {values!r}: {exc}") from exc

    def _option_items(self, options: dict | None) -> list[tuple[tuple[str, ...], Any]]:
        items: list[tuple[tuple[str, ...], Any]] = []

        def visit(prefix: tuple[str, ...], value: Any) -> None:
            if value is None:
                return
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    visit(prefix + tuple(str(nested_key).split(".")), nested_value)
            else:
                items.append((prefix, value))

        for key, value in (options or {}).items():
            visit(tuple(str(key).split(".")), value)
        return items

    def _group_and_key(self, parts: tuple[str, ...]) -> tuple[str, str]:
        if len(parts) == 1:
            return "", parts[0]
        group = parts[0].replace("_", "").replace("-", "").lower()
        return group, parts[-1]

    def _route_two_view_option(
        self, values: dict[str, Any], group: str, key: str, value: Any
    ) -> None:
        if group in {"ransac"} or key in _RANSAC_KEYS:
            self._put_nested(values, ("ransac", key), value)
        elif group in {"twoviewgeometry", "verification"} or key in _TWO_VIEW_GEOMETRY_KEYS:
            values[key] = value
        elif key == "max_error":
            self._put_nested(values, ("ransac", "max_error"), value)

    def _put_nested(self, target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
        current = target
        for part in path[:-1]:
            current = current.setdefault(part, {})
        current[path[-1]] = value

    def _truthy(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on", "single", "single_camera"}

    def _num_reg_images(self, reconstruction: Any) -> int:
        value = getattr(reconstruction, "num_reg_images", 0)
        return int(value() if callable(value) else value)

    def _num_points3d(self, reconstruction: Any) -> int:
        value = getattr(reconstruction, "num_points3D", None)
        if value is not None:
            return int(value() if callable(value) else value)
        points = getattr(reconstruction, "points3D", {})
        return len(points() if callable(points) else points)

    def _colmap_source_version(self) -> str:
        pyproject = REPO_ROOT / "third_party" / "colmap" / "pyproject.toml"
        if not pyproject.exists():
            return "missing"
        for line in pyproject.read_text(encoding="utf-8").splitlines():
            if line.startswith("version = "):
                return line.split("=", 1)[1].strip().strip('"')
        return "unknown"


_IMAGE_READER_KEYS = {
    "camera_model",
    "camera_params",
    "mask_path",
    "camera_mask_path",
    "existing_camera_id",
    "default_focal_length_factor",
}
_FEATURE_EXTRACTION_KEYS = {"type", "max_image_size", "num_threads", "use_gpu", "gpu_index"}
_SIFT_EXTRACTION_KEYS = {
    "max_num_features",
    "first_octave",
    "num_octaves",
    "octave_resolution",
    "peak_threshold",
    "edge_threshold",
    "estimate_affine_shape",
    "max_num_orientations",
    "upright",
    "darkness_adaptivity",
    "domain_size_pooling",
    "dsp_min_scale",
    "dsp_max_scale",
    "dsp_num_scales",
    "normalization",
}
_FEATURE_MATCHING_KEYS = {
    "type",
    "num_threads",
    "use_gpu",
    "gpu_index",
    "max_num_matches",
    "guided_matching",
    "skip_geometric_verification",
    "rig_verification",
    "skip_image_pairs_in_same_frame",
}
_SIFT_MATCHING_KEYS = {"max_ratio", "max_distance", "cross_check", "cpu_brute_force_matcher"}
_PAIRING_KEYS = {
    "exhaustive_matcher": {"block_size"},
    "sequential_matcher": {
        "overlap",
        "quadratic_overlap",
        "expand_rig_images",
        "loop_detection",
        "loop_detection_period",
        "loop_detection_num_images",
        "loop_detection_num_nearest_neighbors",
        "loop_detection_num_checks",
        "loop_detection_num_images_after_verification",
        "loop_detection_max_num_features",
        "vocab_tree_path",
        "num_threads",
    },
    "spatial_matcher": {
        "ignore_z",
        "max_num_neighbors",
        "min_num_neighbors",
        "max_distance",
        "num_threads",
    },
    "vocab_tree_matcher": {
        "num_images",
        "num_nearest_neighbors",
        "num_checks",
        "num_images_after_verification",
        "max_num_features",
        "vocab_tree_path",
        "match_list_path",
        "num_threads",
    },
}
_RANSAC_KEYS = {
    "max_error",
    "min_inlier_ratio",
    "confidence",
    "dyn_num_trials_multiplier",
    "min_num_trials",
    "max_num_trials",
    "random_seed",
    "num_threads",
}
_TWO_VIEW_GEOMETRY_KEYS = {
    "min_num_inliers",
    "min_inlier_ratio",
    "min_E_F_inlier_ratio",
    "max_H_inlier_ratio",
    "watermark_min_inlier_ratio",
    "watermark_border_size",
    "detect_watermark",
    "multiple_ignore_watermark",
    "watermark_detection_max_error",
    "filter_stationary_matches",
    "stationary_matches_max_error",
    "force_H_use",
    "compute_relative_pose",
    "multiple_models",
}
_INCREMENTAL_PIPELINE_KEYS = {
    "min_num_matches",
    "ignore_watermarks",
    "multiple_models",
    "max_num_models",
    "max_model_overlap",
    "min_model_size",
    "init_image_id1",
    "init_image_id2",
    "init_num_trials",
    "extract_colors",
    "num_threads",
    "random_seed",
    "ba_refine_focal_length",
    "ba_refine_principal_point",
    "ba_refine_extra_params",
    "ba_refine_sensor_from_rig",
    "ba_use_gpu",
    "ba_gpu_index",
    "max_runtime_seconds",
}
_INCREMENTAL_MAPPER_KEYS = {
    "init_min_num_inliers",
    "init_max_error",
    "init_max_forward_motion",
    "init_min_tri_angle",
    "abs_pose_max_error",
    "abs_pose_min_num_inliers",
    "abs_pose_min_inlier_ratio",
    "ba_local_num_images",
    "ba_local_min_tri_angle",
    "filter_max_reproj_error",
    "filter_min_tri_angle",
    "max_reg_trials",
}
_GLOBAL_PIPELINE_KEYS = {
    "random_seed",
    "num_threads",
    "skip_rotation_averaging",
    "skip_track_establishment",
    "skip_global_positioning",
    "skip_bundle_adjustment",
    "skip_retriangulation",
}
_BUNDLE_ADJUSTMENT_KEYS = {
    "refine_focal_length",
    "refine_principal_point",
    "refine_extra_params",
    "refine_rig_from_world",
    "refine_sensor_from_rig",
    "constant_rig_from_world_rotation",
    "refine_points3D",
    "min_track_length",
    "print_summary",
    "backend",
}
_CERES_BA_KEYS = {
    "loss_function_type",
    "loss_function_scale",
    "use_gpu",
    "gpu_index",
    "min_num_images_gpu_solver",
    "min_num_residuals_for_cpu_multi_threading",
    "max_num_images_direct_dense_cpu_solver",
    "max_num_images_direct_sparse_cpu_solver",
    "max_num_images_direct_dense_gpu_solver",
    "max_num_images_direct_sparse_gpu_solver",
    "auto_select_solver_type",
}
