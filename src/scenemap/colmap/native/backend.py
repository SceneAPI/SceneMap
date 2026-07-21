from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import struct
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..io_mapper import ColmapMapper
from ..model import Reconstruction, read_colmap_text_model

try:
    from sceneapi.errors import CapabilityUnavailableError, ValidationError
except ModuleNotFoundError:  # pragma: no cover - only for package metadata tools

    class CapabilityUnavailableError(RuntimeError):  # type: ignore[no-redef]
        def __init__(self, *, capability: str, reason: str = "") -> None:
            super().__init__(reason or capability)

    class ValidationError(RuntimeError):  # type: ignore[no-redef]
        pass


# parents[4]: backend.py lives two package levels deeper than in the
# superseded per-provider repos (src/scenemap/colmap/<provider>/backend.py).
REPO_ROOT = Path(__file__).resolve().parents[4]
COLMAP_PAIR_ID_BASE = 2_147_483_647
_BRACKET_PROGRESS_RE = re.compile(r"\[(\d+)\s*/\s*(\d+)\]")
_PERCENT_PROGRESS_RE = re.compile(r"(?<!\d)(100|[1-9]?\d)(?:\.\d+)?%")


def colmap_runtime_path_dirs(executable: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if executable is not None:
        candidates.append(executable.parent)

    if os.name == "nt":
        raw_dirs = os.environ.get("SFMAPI_COLMAP_DLL_DIRS", "")
        candidates.extend(Path(item) for item in raw_dirs.split(os.pathsep) if item)

        project_root = REPO_ROOT.parent
        candidates.extend(
            [
                REPO_ROOT / "third_party" / "colmap" / "install" / "bin",
                project_root / "colmap-install-cuda-cudss" / "bin",
                project_root / "vcpkg_installed_colmap_cuda" / "x64-windows" / "bin",
                project_root / "vcpkg_installed_colmap" / "x64-windows" / "bin",
            ]
        )

        cuda_root = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
        if cuda_root:
            candidates.extend([Path(cuda_root) / "bin", Path(cuda_root) / "bin" / "x64"])

        cudss_root = os.environ.get("CUDSS_ROOT") or os.environ.get("CUDSS_PATH")
        if cudss_root:
            candidates.extend([Path(cudss_root) / "bin", Path(cudss_root) / "bin" / "13"])
        candidates.extend(_default_cudss_bin_dirs())

    resolved_dirs: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        resolved_dirs.append(resolved)
    return resolved_dirs


def colmap_runtime_env(executable: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    dirs = colmap_runtime_path_dirs(executable)
    if dirs:
        env["PATH"] = os.pathsep.join([*(str(path) for path in dirs), env.get("PATH", "")])
    return env


def _default_cudss_bin_dirs() -> list[Path]:
    root = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "NVIDIA cuDSS"
    if not root.is_dir():
        return []
    return sorted({path.parent for path in root.glob(r"v*\bin\*\cudss64_*.dll")})


def _plugin_cache_root(plugin_id: str) -> Path:
    override = os.environ.get("SFMAPI_PLUGIN_CACHE")
    if override:
        return Path(os.path.expandvars(override)).expanduser() / plugin_id
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "sfmapi" / "plugins" / plugin_id
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "sfmapi" / "plugins" / plugin_id


def _cached_colmap_candidates() -> list[Path]:
    cache = _plugin_cache_root("colmap")
    names = ["colmap.exe", "colmap.bat", "colmap"] if os.name == "nt" else ["colmap"]
    candidates: list[Path] = []
    for name in names:
        candidates.extend([cache / "current" / name, cache / "current" / "bin" / name])
    if cache.exists():
        for child in cache.iterdir():
            if child.is_dir() and child.name != "current":
                for name in names:
                    candidates.extend([child / name, child / "bin" / name])
    return candidates


@dataclass
class TwoViewGeometryRow:
    config: int
    inlier_matches: list[tuple[int, int]]
    F: list[list[float]] | None = None
    E: list[list[float]] | None = None
    H: list[list[float]] | None = None
    qvec: list[float] | None = None
    tvec: list[float] | None = None

    @property
    def num_inliers(self) -> int:
        return len(self.inlier_matches)


COLMAP_EXPORT_TYPES: dict[str, str] = {
    "colmap_text": "TXT",
    "txt": "TXT",
    "colmap_bin": "BIN",
    "bin": "BIN",
    "ply": "PLY",
    "nvm": "NVM",
    "bundler": "Bundler",
    "vrml": "VRML",
    "r3d": "R3D",
    "cam": "CAM",
}

# COLMAP CLI subcommand surface. PINNED-VERSION ASSUMPTION: these
# names track the COLMAP 3.11.x command surface (see ``compatibility``
# in plugin.py — ``colmap >=3.9``, exercised against 3.11.1 sample
# data). COLMAP occasionally renames subcommands across releases (the
# global-mapping entry point in particular has moved); this backend
# does NOT probe ``colmap help`` to adapt. If a newer COLMAP renames a
# command, the affected stage fails with a clear "unknown command"
# error rather than silently misbehaving — update this tuple +
# ``COLMAP_BACKEND_CONFIGS`` + ``MATCH_COMMANDS`` to support a new major.
COLMAP_COMMANDS: tuple[str, ...] = (
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
    "geometric_verifier",
    "global_mapper",
    "guided_geometric_verifier",
    "help",
    "version",
    "hierarchical_mapper",
    "image_deleter",
    "image_filterer",
    "image_rectifier",
    "image_registrator",
    "image_undistorter",
    "image_undistorter_standalone",
    "mapper",
    "matches_importer",
    "mesh_simplifier",
    "mesh_texturer",
    "model_aligner",
    "model_analyzer",
    "model_clusterer",
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
    "pose_prior_mapper",
    "poisson_mesher",
    "project_generator",
    "rig_configurator",
    "rotation_averager",
    "sequential_matcher",
    "spatial_matcher",
    "stereo_fusion",
    "transitive_matcher",
    "view_graph_calibrator",
    "vocab_tree_builder",
    "vocab_tree_matcher",
    "vocab_tree_retriever",
)

MATCH_COMMANDS: dict[str, str] = {
    "exhaustive": "exhaustive_matcher",
    "sequential": "sequential_matcher",
    "spatial": "spatial_matcher",
    # ``from_poses`` selects pairs by camera-position proximity. COLMAP
    # has no dedicated command for it: spatial_matcher already pairs
    # images by their location priors, so it is the portable backing
    # command for the ``pairs.from_poses`` capability.
    "from_poses": "spatial_matcher",
    "vocabtree": "vocab_tree_matcher",
    "vocab_tree": "vocab_tree_matcher",
    "explicit": "matches_importer",
    "transitive": "transitive_matcher",
}

# Single source of truth: the canonical COLMAP stage-config table + the
# runtime-managed CLI option set are COLMAP vendor data, owned by the
# plugin-local stage_configs module and shared by all three COLMAP-family
# providers so their served config schemas cannot drift. MATCH_COMMANDS above
# stays plugin-local (command routing, not config schemas).
from ..stage_configs import COLMAP_STAGE_CONFIGS as COLMAP_BACKEND_CONFIGS  # noqa: E402
from ..stage_configs import RUNTIME_MANAGED_COLMAP_OPTIONS  # noqa: E402

_COLMAP_OPTION_LINE_RE = re.compile(
    r"^\s*(?:(?P<short>-\w)\s+\[\s*(?P<long_alias>--[\w.\-]+)\s*\]|"
    r"(?P<long>--[\w.\-]+))(?P<rest>.*)$"
)
_COLMAP_DEFAULT_RE = re.compile(r"\(=([^)]+)\)")
_COLMAP_CHOICES_RE = re.compile(r"\{([^{}]*)\}")


def _parse_colmap_command_help(command: str, help_text: str) -> dict[str, Any]:
    options: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in help_text.splitlines():
        parsed = _parse_option_start(line)
        if parsed is not None:
            if current is not None:
                options.append(_finalize_option_schema(current))
            current = parsed
            continue

        if current is not None:
            stripped = line.strip()
            if stripped:
                current["_tail_parts"].append(stripped)

    if current is not None:
        options.append(_finalize_option_schema(current))

    return {
        "command": command,
        "available": True,
        "schema_source": "colmap_help",
        "option_count": len(options),
        "options": options,
    }


def _parse_option_start(line: str) -> dict[str, Any] | None:
    match = _COLMAP_OPTION_LINE_RE.match(line)
    if match is None:
        return None

    short = match.group("short")
    long_flag = match.group("long_alias") or match.group("long")
    rest = (match.group("rest") or "").strip()
    value_name = None
    if rest == "arg" or rest.startswith("arg "):
        value_name = "arg"
        rest = rest[3:].strip()

    flags = [flag for flag in (short, long_flag) if flag]
    name = (long_flag or short or "").lstrip("-")
    return {
        "name": name,
        "flags": flags,
        "value_name": value_name,
        "_tail_parts": [rest] if rest else [],
    }


def _finalize_option_schema(option: dict[str, Any]) -> dict[str, Any]:
    tail = " ".join(str(part) for part in option.pop("_tail_parts", []))
    tail = re.sub(r"\s+", " ", tail).strip()

    default_raw = None
    default_match = _COLMAP_DEFAULT_RE.search(tail)
    if default_match is not None:
        default_raw = default_match.group(1)
        tail = _COLMAP_DEFAULT_RE.sub("", tail, count=1).strip()

    choices: list[str] = []
    for raw_choices in _COLMAP_CHOICES_RE.findall(tail):
        choices.extend(
            choice.strip().strip("'\"") for choice in raw_choices.split(",") if choice.strip()
        )
    tail = _COLMAP_CHOICES_RE.sub("", tail).strip()
    description = re.sub(r"\s+", " ", tail).strip()

    option_type, option_format = _infer_option_schema_type(
        str(option["name"]),
        option.get("value_name"),
        default_raw,
        choices,
    )
    default = _parse_option_default(default_raw, option_type)
    schema: dict[str, Any] = {"type": option_type}
    if option_format:
        schema["format"] = option_format
    if choices:
        schema["enum"] = choices
    if default_raw is not None:
        schema["default"] = default
    if description:
        schema["description"] = description

    return {
        "name": option["name"],
        "flags": option["flags"],
        "takes_value": option.get("value_name") is not None,
        "value_name": option.get("value_name"),
        "type": option_type,
        "format": option_format,
        "default": default,
        "default_raw": default_raw,
        "choices": choices,
        "description": description,
        "required": None,
        "schema": schema,
    }


def _infer_option_schema_type(
    name: str,
    value_name: str | None,
    default_raw: str | None,
    choices: list[str],
) -> tuple[str, str | None]:
    normalized_name = name.lower()
    if value_name is None:
        return "boolean", None
    if choices:
        return "string", None
    if default_raw is not None:
        if re.fullmatch(r"-?\d+", default_raw):
            return "integer", None
        if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)(?:e[+-]?\d+)?", default_raw, re.IGNORECASE):
            return "number", None
    if normalized_name.endswith("_path") or normalized_name.endswith("_paths"):
        return "string", "path"
    if normalized_name.endswith("_dir") or normalized_name.endswith("_directory"):
        return "string", "path"
    return "string", None


def _parse_option_default(default_raw: str | None, option_type: str) -> Any:
    if default_raw is None:
        return None
    if option_type == "integer":
        try:
            return int(default_raw)
        except ValueError:
            return default_raw
    if option_type == "number":
        try:
            return float(default_raw)
        except ValueError:
            return default_raw
    if option_type == "boolean":
        return default_raw.lower() in {"1", "true", "yes", "on"}
    return default_raw


def _fallback_colmap_command_schema(command: str, reason: str) -> dict[str, Any] | None:
    if command != "delaunay_mesher":
        return None

    options = [
        *_common_colmap_cli_options(),
        _manual_option(
            "input_path",
            option_format="path",
            description="Path to either the dense workspace folder or the sparse reconstruction",
            required=True,
        ),
        _manual_option(
            "input_type",
            default="dense",
            choices=["dense", "sparse"],
            description="Input representation to mesh.",
        ),
        _manual_option("output_path", option_format="path", required=True),
        _manual_option("DelaunayMeshing.max_proj_dist", default=20.0),
        _manual_option("DelaunayMeshing.max_depth_dist", default=0.05),
        _manual_option("DelaunayMeshing.visibility_sigma", default=3.0),
        _manual_option("DelaunayMeshing.distance_sigma_factor", default=1.0),
        _manual_option("DelaunayMeshing.quality_regularization", default=1.0),
        _manual_option("DelaunayMeshing.max_side_length_factor", default=25.0),
        _manual_option("DelaunayMeshing.max_side_length_percentile", default=95.0),
        _manual_option("DelaunayMeshing.num_threads", default=-1),
    ]
    return {
        "command": command,
        "available": False,
        "schema_source": "colmap_source_fallback",
        "unavailable_reason": reason,
        "option_count": len(options),
        "options": options,
    }


def _common_colmap_cli_options() -> list[dict[str, Any]]:
    return [
        _manual_option("help", flags=["-h", "--help"], takes_value=False, option_type="boolean"),
        _manual_option("project_path", option_format="path"),
        _manual_option("default_random_seed", default=0),
        _manual_option(
            "log_target",
            default="stderr_and_file",
            choices=["stderr", "stdout", "file", "stderr_and_file"],
        ),
        _manual_option("log_path", option_format="path"),
        _manual_option("log_level", default=0),
        _manual_option(
            "log_severity",
            default=0,
            description="0:INFO, 1:WARNING, 2:ERROR, 3:FATAL",
        ),
        _manual_option("log_color", default=1),
    ]


def _manual_option(
    name: str,
    *,
    flags: list[str] | None = None,
    takes_value: bool = True,
    option_type: str | None = None,
    option_format: str | None = None,
    default: Any = None,
    choices: list[str] | None = None,
    description: str = "",
    required: bool | None = None,
) -> dict[str, Any]:
    inferred_type = option_type or _manual_option_type(default, choices)
    schema: dict[str, Any] = {"type": inferred_type}
    if option_format:
        schema["format"] = option_format
    if choices:
        schema["enum"] = choices
    if default is not None:
        schema["default"] = default
    if description:
        schema["description"] = description

    return {
        "name": name,
        "flags": flags or [f"--{name}"],
        "takes_value": takes_value,
        "value_name": "arg" if takes_value else None,
        "type": inferred_type,
        "format": option_format,
        "default": default,
        "default_raw": str(default) if default is not None else None,
        "choices": choices or [],
        "description": description,
        "required": required,
        "schema": schema,
    }


def _manual_option_type(default: Any, choices: list[str] | None) -> str:
    if choices:
        return "string"
    if isinstance(default, bool):
        return "boolean"
    if isinstance(default, int):
        return "integer"
    if isinstance(default, float):
        return "number"
    return "string"


class ColmapCliBackend(ColmapMapper):
    """sfmapi backend that shells out to the upstream COLMAP CLI."""

    name = "colmap_cli"
    version = "0.0.1"
    vendor = "COLMAP upstream"

    def __init__(self, executable: str | Path | None = None) -> None:
        self._executable_override = Path(executable) if executable else None
        self._command_schema_cache: dict[str, dict[str, Any]] = {}

    def capabilities(self) -> set[str]:
        if self._find_colmap() is None:
            return set()
        return {
            "features.extract.sift",
            "matches.verify",
            "pairs.exhaustive",
            "pairs.sequential",
            "pairs.spatial",
            "pairs.from_poses",
            "pairs.vocabtree",
            "pairs.explicit",
            "matchers.nn-mutual",
            "matchers.nn-ratio",
            "map.incremental",
            "map.global",
            "map.hierarchical",
            "ba.standard",
            "triangulate.retri",
            "relocalize.images",
            "pgo.optimize",
            "recon.merge",
            "export.ply",
            "export.nvm",
            "export.colmap_text",
            "export.colmap_bin",
            "georegister.sim3",
            "georegister.gps",
            "image.undistort",
            "index.vocab_tree",
            "rigs.configure",
            "pose_priors.mapping",
        }

    def list_backend_actions(self, *, include_schemas: bool = False) -> list[dict[str, Any]]:
        """Expose COLMAP commands through sfmapi's backend-action catalog.

        These are backend-native extension actions, not portable
        sfmapi capabilities. Keeping them here lets sfmapi expose every
        COLMAP command and schema without polluting ``/v1/capabilities``.
        """
        if self._find_colmap() is None:
            return []
        return [
            self._backend_action_descriptor(
                command,
                schema=self.colmap_command_schema(command) if include_schemas else None,
            )
            for command in COLMAP_COMMANDS
        ]

    def list_backend_config_schemas(self, *, include_schemas: bool = True) -> list[dict[str, Any]]:
        """Expose COLMAP stage option schemas for sfmapi ``backend_options``.

        The action catalog covers arbitrary COLMAP commands. This
        catalog is narrower: it describes provider-specific options
        that can be passed to portable sfmapi stages such as feature
        extraction, pair selection, matching, verification, mapping,
        and bundle adjustment.
        """
        if self._find_colmap() is None:
            return []
        capabilities = self.capabilities()
        rows: list[dict[str, Any]] = []
        for config_id, stage, capability, provider, command in COLMAP_BACKEND_CONFIGS:
            if capability not in capabilities:
                continue
            schema = None
            if include_schemas:
                try:
                    schema = self.colmap_command_schema(command)
                except Exception:
                    # This COLMAP build lacks the command (e.g. older builds
                    # without global_mapper); skip the one config rather than
                    # failing the whole listing (matches the framework guard).
                    continue
            rows.append(
                self._backend_config_descriptor(
                    config_id=config_id,
                    stage=stage,
                    capability=capability,
                    provider=provider,
                    command=command,
                    schema=schema,
                )
            )
        return rows

    def get_backend_action(self, action_id: str) -> dict[str, Any]:
        command = self._backend_action_command(action_id)
        return self._backend_action_descriptor(
            command,
            schema=self.colmap_command_schema(command),
        )

    def validate_backend_action(self, action_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
        command = self._backend_action_command(action_id)
        options, positional = self._split_backend_action_inputs(inputs)
        try:
            self._validated_colmap_option_args(command, options)
        except ValidationError as exc:
            return {
                "action_id": action_id,
                "valid": False,
                "errors": [{"field": None, "message": str(exc)}],
                "normalized_inputs": {},
            }
        normalized_inputs: dict[str, Any] = dict(options)
        if positional:
            normalized_inputs["positional_args"] = positional
        return {
            "action_id": action_id,
            "valid": True,
            "errors": [],
            "normalized_inputs": normalized_inputs,
        }

    def run_backend_action(
        self,
        action_id: str,
        inputs: dict[str, Any],
        **_: Any,
    ) -> dict[str, Any]:
        command = self._backend_action_command(action_id)
        options, positional = self._split_backend_action_inputs(inputs)
        return self.run_colmap_command(command, options=options, positional=positional)

    def extract_features(
        self,
        *,
        database_path: Path,
        image_root: Path,
        image_list: list[str],
        options: dict,
        progress: Any | None = None,
    ) -> dict:
        exe = self._require_colmap("features.extract.sift")
        database_path.parent.mkdir(parents=True, exist_ok=True)
        image_root = Path(image_root)

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
            list_path = Path(tmp.name)
            for image in image_list:
                tmp.write(f"{image}\n")
        try:
            args = [
                exe,
                "feature_extractor",
                "--database_path",
                str(database_path),
                "--image_path",
                str(image_root),
                "--image_list_path",
                str(list_path),
            ]
            self._append_feature_options(args, options)
            total = len(image_list)
            self._progress(progress, "feature_extraction", current=0, total=total)
            self._run_stage(
                args,
                progress=progress,
                progress_phase="feature_extraction",
                progress_total=total,
            )
            self._progress(progress, "feature_extraction", current=total, total=total)
        finally:
            list_path.unlink(missing_ok=True)

        return {
            "num_images": len(image_list),
            "database_path": str(database_path),
            "engine": "colmap feature_extractor",
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
        command = MATCH_COMMANDS.get(normalized_mode)
        if command is None:
            raise CapabilityUnavailableError(
                capability="backend.actions",
                reason=(
                    "upstream COLMAP CLI demo wires exhaustive, sequential, "
                    "spatial, vocabtree, and transitive matching"
                ),
            )
        cleanup_path: Path | None = None
        if command == "matches_importer":
            options, cleanup_path = self._explicit_match_options(options)
        exe = self._require_colmap(self._match_capability(normalized_mode))
        try:
            args = [exe, command, "--database_path", str(database_path)]
            self._append_match_options(args, command, options)
            total = self._match_progress_total(database_path, normalized_mode, options)
            self._progress(progress, "matching", current=0, total=total)
            self._run_stage(
                args, progress=progress, progress_phase="matching", progress_total=total
            )
            if total is not None:
                self._progress(progress, "matching", current=total, total=total)
        finally:
            if cleanup_path is not None:
                cleanup_path.unlink(missing_ok=True)
        return {
            "database_path": str(database_path),
            "strategy": mode,
            "engine": f"colmap {command}",
        }

    def verify_matches(
        self,
        *,
        database_path: Path,
        options: dict,
        progress: Any | None = None,
    ) -> dict:
        exe = self._require_colmap("matches.verify")
        args = [exe, "geometric_verifier", "--database_path", str(database_path)]
        self._append_verify_options(args, options)
        total = self._database_match_pair_count(database_path)
        self._progress(progress, "geometric_verification", current=0, total=total)
        self._run_stage(
            args,
            progress=progress,
            progress_phase="geometric_verification",
            progress_total=total,
        )
        if total is not None:
            self._progress(progress, "geometric_verification", current=total, total=total)
        return {
            "database_path": str(database_path),
            "num_verified_pairs": sum(
                1 for _ in self.iter_two_view_geometries(database_path=database_path)
            ),
            "engine": "colmap geometric_verifier",
        }

    def read_keypoints(
        self,
        *,
        database_path: Path,
        image_id: int,
    ) -> tuple[list[list[float]], bytes, int]:
        with sqlite3.connect(database_path) as conn:
            row = conn.execute(
                "select rows, cols, data from keypoints where image_id = ?",
                (int(image_id),),
            ).fetchone()
            if row is None:
                return [], b"", 128
            keypoint_rows, keypoint_cols, keypoint_blob = row
            descriptor_row = conn.execute(
                "select rows, cols, data from descriptors where image_id = ?",
                (int(image_id),),
            ).fetchone()

        keypoints = self._decode_keypoints(
            rows=int(keypoint_rows),
            cols=int(keypoint_cols),
            blob=bytes(keypoint_blob or b""),
        )
        if descriptor_row is None:
            return keypoints, b"", 0
        descriptor_rows, descriptor_cols, descriptor_blob = descriptor_row
        descriptors = self._decode_descriptor_bytes(
            rows=int(descriptor_rows),
            cols=int(descriptor_cols),
            blob=bytes(descriptor_blob or b""),
        )
        return keypoints, descriptors, int(descriptor_cols)

    def iter_two_view_geometries(self, *, database_path: Path) -> Iterator[tuple[int, int, Any]]:
        with sqlite3.connect(database_path) as conn:
            geometry_rows = conn.execute(
                "select pair_id, rows, cols, data, config, F, E, H, qvec, tvec "
                "from two_view_geometries"
            ).fetchall()
        for (
            pair_id,
            num_rows,
            cols,
            data,
            config,
            f_blob,
            e_blob,
            h_blob,
            q_blob,
            t_blob,
        ) in geometry_rows:
            image_id1, image_id2 = self._pair_id_to_image_ids(int(pair_id))
            geom = TwoViewGeometryRow(
                config=int(config or 0),
                inlier_matches=self._decode_uint_matrix(int(num_rows or 0), int(cols or 0), data),
                F=self._decode_matrix_3x3(f_blob),
                E=self._decode_matrix_3x3(e_blob),
                H=self._decode_matrix_3x3(h_blob),
                qvec=self._decode_float64_vector(q_blob),
                tvec=self._decode_float64_vector(t_blob),
            )
            yield image_id1, image_id2, geom

    def iter_correspondences(self, *, database_path: Path) -> Iterator[tuple[int, int, Any]]:
        with sqlite3.connect(database_path) as conn:
            match_rows = conn.execute("select pair_id, rows, cols, data from matches").fetchall()
        for pair_id, num_rows, cols, data in match_rows:
            matches = self._decode_uint_matrix(int(num_rows or 0), int(cols or 0), data)
            if not matches:
                continue
            image_id1, image_id2 = self._pair_id_to_image_ids(int(pair_id))
            yield image_id1, image_id2, matches

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
    ) -> tuple[list[dict], list[Reconstruction]]:
        normalized_kind = kind.replace("-", "_").lower()
        command = {
            "global": "global_mapper",
            "glomap": "global_mapper",
            "incremental": "mapper",
            "hierarchical": "hierarchical_mapper",
        }.get(normalized_kind)
        if command is None:
            raise CapabilityUnavailableError(capability=f"map.{kind}")
        # COLMAP consumes per-image position priors through the
        # dedicated ``pose_prior_mapper`` command, which reads the
        # SQLite ``pose_priors`` table. sfmapi keeps priors in its own
        # store, so when the worker hands us priors we materialize them
        # into the COLMAP feature DB and swap ``mapper`` for the
        # pose-prior-aware mapper. Global / hierarchical mapping have no
        # pose-prior variant, so priors are silently ignored there.
        use_pose_priors = bool(pose_priors) and command == "mapper"
        if use_pose_priors:
            command = "pose_prior_mapper"
            self._write_pose_priors(db_path, pose_priors or {})
        exe = self._require_colmap(f"map.{normalized_kind}")
        sparse_root.mkdir(parents=True, exist_ok=True)
        job_dir.mkdir(parents=True, exist_ok=True)

        args = [
            exe,
            command,
            "--database_path",
            str(db_path),
            "--image_path",
            str(image_root),
            "--output_path",
            str(sparse_root),
        ]
        self._append_mapper_options(args, spec, command)
        phase = self._mapping_progress_phase(normalized_kind)
        total = self._database_image_count(db_path)
        self._progress(progress, phase, current=0, total=total)
        self._run_stage(args, progress=progress, progress_phase=phase, progress_total=total)

        summaries: list[dict] = []
        reconstructions: list[Reconstruction] = []
        model_dirs = [p for p in sorted(sparse_root.iterdir()) if p.is_dir()]
        if not model_dirs and any(
            (sparse_root / name).exists()
            for name in ("cameras.bin", "cameras.txt", "images.bin", "images.txt")
        ):
            model_dirs = [sparse_root]
        for model_dir in model_dirs:
            model_name = model_dir.name if model_dir != sparse_root else "0"
            text_dir = job_dir / "colmap_text_models" / model_name
            text_dir.mkdir(parents=True, exist_ok=True)
            self._convert_model(model_dir, text_dir, "TXT")
            rec = read_colmap_text_model(text_dir)
            reconstructions.append(rec)
            summaries.append(
                {
                    "idx": int(model_name) if model_name.isdigit() else model_name,
                    "num_reg_images": rec.num_reg_images(),
                    "num_points3D": len(rec.points3D),
                    "model_path": str(model_dir),
                }
            )
        if total is not None:
            self._progress(progress, phase, current=total, total=total)
        return summaries, reconstructions

    def bundle_adjustment(self, *, model_path: Path, output_path: Path, spec: dict) -> dict:
        exe = self._require_colmap("ba.standard")
        output_path.mkdir(parents=True, exist_ok=True)
        options = spec.get("bundle_adjustment") or spec.get("options") or spec
        args = [
            exe,
            "bundle_adjuster",
            "--input_path",
            str(model_path),
            "--output_path",
            str(output_path),
        ]
        self._append_options(args, "BundleAdjustment", options)
        self._run(args)
        rec = self.read_reconstruction(output_path)
        return {
            "model_path": str(output_path),
            "mode": spec.get("mode", "standard"),
            "num_reg_images": rec.num_reg_images(),
            "num_points3D": len(rec.points3D),
            "engine": "colmap bundle_adjuster",
        }

    def triangulate(
        self,
        *,
        model_path: Path,
        database_path: Path,
        image_root: Path,
        output_path: Path,
    ) -> dict:
        exe = self._require_colmap("triangulate.retri")
        output_path.mkdir(parents=True, exist_ok=True)
        self._run(
            [
                exe,
                "point_triangulator",
                "--database_path",
                str(database_path),
                "--image_path",
                str(image_root),
                "--input_path",
                str(model_path),
                "--output_path",
                str(output_path),
            ]
        )
        rec = self.read_reconstruction(output_path)
        return {
            "model_path": str(output_path),
            "num_reg_images": rec.num_reg_images(),
            "num_points3D": len(rec.points3D),
            "engine": "colmap point_triangulator",
        }

    def relocalize(
        self,
        *,
        model_path: Path,
        database_path: Path,
        image_root: Path,
        output_path: Path,
        image_ids: list[int],
    ) -> dict:
        exe = self._require_colmap("relocalize.images")
        output_path.mkdir(parents=True, exist_ok=True)
        self._run(
            [
                exe,
                "image_registrator",
                "--database_path",
                str(database_path),
                "--input_path",
                str(model_path),
                "--output_path",
                str(output_path),
            ]
        )
        rec = self.read_reconstruction(output_path)
        return {
            "model_path": str(output_path),
            "requested_image_ids": [int(x) for x in image_ids],
            "num_reg_images": rec.num_reg_images(),
            "num_points3D": len(rec.points3D),
            "engine": "colmap image_registrator",
        }

    def pose_graph_optimize(self, *, model_path: Path, output_path: Path, spec: dict) -> dict:
        """Run pose-graph optimization via COLMAP ``rotation_averager``.

        COLMAP exposes global rotation averaging over a reconstruction's
        view graph as the ``rotation_averager`` command (``--input_path``
        / ``--output_path``). That is the closest portable analogue to
        sfmapi's ``pgo.optimize`` stage, so this wraps it directly.
        Backend-specific ``rotation_averager`` options ride in via
        ``spec`` (``backend_options`` or a flat dict) and are appended
        with no COLMAP option-group prefix.
        """
        exe = self._require_colmap("pgo.optimize")
        output_path.mkdir(parents=True, exist_ok=True)
        args = [
            exe,
            "rotation_averager",
            "--input_path",
            str(model_path),
            "--output_path",
            str(output_path),
        ]
        options = spec.get("pose_graph_optimization") or spec.get("options") or spec
        self._append_rotation_averager_options(args, options)
        self._run(args)
        rec = self.read_reconstruction(output_path)
        return {
            "model_path": str(output_path),
            "mode": spec.get("mode", "rotation_averaging"),
            "num_reg_images": rec.num_reg_images(),
            "num_points3D": len(rec.points3D),
            "engine": "colmap rotation_averager",
        }

    def export(self, *, model_path: Path, output_path: Path, format: str) -> dict:
        format_key = format.replace("-", "_").lower()
        output_type = COLMAP_EXPORT_TYPES.get(format_key)
        if output_type is None:
            raise CapabilityUnavailableError(capability=f"export.{format}")
        self._require_colmap(f"export.{format_key}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_type in {"TXT", "BIN"}:
            output_path.mkdir(parents=True, exist_ok=True)
        self._convert_model(model_path, output_path, output_type)
        return {"format": format_key, "output_path": str(output_path)}

    def merge_reconstructions(
        self,
        *,
        model_paths: list[Path],
        output_path: Path,
        sim3_aligners: Any = None,
    ) -> dict:
        """Merge N sparse models into one via COLMAP ``model_merger``.

        COLMAP's ``model_merger`` is pairwise (``--input_path1`` +
        ``--input_path2``), so this folds the input list left-to-right:
        merge[0,1] -> merge[result,2] -> ... The optional
        ``sim3_aligners`` argument is accepted for sfmapi protocol
        parity; COLMAP's pairwise merger estimates the alignment
        itself, so explicit aligners are not forwarded.
        """
        paths = [Path(model_path) for model_path in model_paths]
        if len(paths) < 2:
            raise ValidationError("merge_reconstructions requires at least two model_paths")
        exe = self._require_colmap("recon.merge")
        output_path.mkdir(parents=True, exist_ok=True)

        max_reproj_error: float | None = None
        if isinstance(sim3_aligners, dict):
            raw_error = sim3_aligners.get("max_reproj_error")
            if raw_error is not None:
                max_reproj_error = float(raw_error)

        merged = paths[0]
        with tempfile.TemporaryDirectory() as scratch_root:
            scratch = Path(scratch_root)
            for index, next_path in enumerate(paths[1:]):
                is_last = index == len(paths) - 2
                step_output = output_path if is_last else scratch / f"merge_{index}"
                step_output.mkdir(parents=True, exist_ok=True)
                args = [
                    exe,
                    "model_merger",
                    "--input_path1",
                    str(merged),
                    "--input_path2",
                    str(next_path),
                    "--output_path",
                    str(step_output),
                ]
                if max_reproj_error is not None:
                    args.extend(["--max_reproj_error", str(max_reproj_error)])
                self._run(args)
                merged = step_output

        rec = self.read_reconstruction(output_path)
        return {
            "model_path": str(output_path),
            "num_sources": len(paths),
            "num_reg_images": rec.num_reg_images(),
            "num_points3D": len(rec.points3D),
            "engine": "colmap model_merger",
        }

    def list_backend_artifact_contracts(self) -> list[dict[str, Any]]:
        """Describe the artifact kinds COLMAP stages accept and emit.

        These are honest, backend-specific I/O contracts: COLMAP
        materializes a SQLite ``database.db`` (keypoints, descriptors,
        matches, and two-view geometries) and a sparse reconstruction
        model directory (``cameras``/``images``/``points3D``). The
        ``contract_id`` values are the stable handles sfmapi exposes
        through ``/v1/backend/artifact-contracts``.
        """
        return [
            {
                "contract_id": "colmap.database",
                "stage": "features",
                "capability": "features.extract.sift",
                "display_name": "COLMAP SQLite feature/match database",
                "description": (
                    "COLMAP database.db holding per-image keypoints and "
                    "descriptors plus raw and geometrically verified matches."
                ),
                "accepts": [],
                "emits": ["features.local.v1"],
                "preferred": "features.local.v1",
            },
            {
                "contract_id": "colmap.sparse_model",
                "stage": "mapping",
                "capability": "map.incremental",
                "display_name": "COLMAP sparse reconstruction model",
                "description": (
                    "COLMAP sparse model directory (cameras, image poses, "
                    "tracks, and sparse 3D points) produced by the mapper."
                ),
                "accepts": ["matches.verified.v1"],
                "emits": ["reconstruction.sparse.v1", "reconstruction.snapshot"],
                "preferred": "reconstruction.sparse.v1",
            },
        ]

    def convert_spherical_to_cubemap(
        self,
        *,
        input_model_path: Path,
        input_image_path: Path,
        output_path: Path,
    ) -> dict:
        raise CapabilityUnavailableError(capability="projection.cubemap_rig")

    def render_spherical_cubemap_images(
        self,
        *,
        input_image_path: Path,
        output_path: Path,
        face_size: int | None = None,
    ) -> dict:
        raise CapabilityUnavailableError(capability="projection.equirectangular_to_cubemap")

    def build_vlad_index(
        self,
        *,
        image_paths_by_id: dict[str, Path],
        spec: dict,
    ) -> tuple[list[str], Any]:
        raise CapabilityUnavailableError(capability="similarity.vlad")

    def localize_from_memory(self, *, sparse_dir: Path, query_image: Path, spec: dict) -> dict:
        raise CapabilityUnavailableError(capability="localize.from_memory")

    def apply_sim3(
        self,
        *,
        model_path: Path,
        output_path: Path,
        sim3: dict,
    ) -> dict:
        exe = self._require_colmap("georegister.sim3")
        output_path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
            transform_path = Path(tmp.name)
            tmp.write(self._sim3_to_transform_text(sim3))
        try:
            self._run(
                [
                    exe,
                    "model_transformer",
                    "--input_path",
                    str(model_path),
                    "--output_path",
                    str(output_path),
                    "--transform_path",
                    str(transform_path),
                ]
            )
        finally:
            transform_path.unlink(missing_ok=True)
        rec = self.read_reconstruction(output_path)
        return {
            "model_path": str(output_path),
            "num_reg_images": rec.num_reg_images(),
            "num_points3D": len(rec.points3D),
            "engine": "colmap model_transformer",
        }

    def align_reconstruction(
        self,
        *,
        model_path: Path,
        output_path: Path,
        spec: dict,
    ) -> dict:
        """Georegister a sparse model via COLMAP ``model_aligner``.

        Unlike :meth:`apply_sim3` (which applies a caller-supplied
        Sim(3)), ``model_aligner`` *solves* the alignment transform from
        georeferenced inputs: a reference model, a reference-image
        location file, or GPS priors stored in the database. Inputs are
        declared in ``spec`` — ``ref_images_path`` / ``ref_model_path``
        / ``database_path`` plus ``ref_is_gps``, ``alignment_type``,
        ``min_common_images``, etc. — and forwarded to the COLMAP CLI
        verbatim (no option-group prefix).
        """
        exe = self._require_colmap("georegister.gps")
        output_path.mkdir(parents=True, exist_ok=True)
        args = [
            exe,
            "model_aligner",
            "--input_path",
            str(model_path),
            "--output_path",
            str(output_path),
        ]
        self._append_prefixless_options(
            args,
            spec,
            allowed={
                "database_path",
                "ref_model_path",
                "ref_images_path",
                "ref_is_gps",
                "merge_image_and_ref_origins",
                "transform_path",
                "alignment_type",
                "min_common_images",
                "alignment_max_error",
                "estimate_scale",
                "robust_alignment",
                "robust_alignment_max_error",
                "num_threads",
            },
        )
        self._run(args)
        rec = self.read_reconstruction(output_path)
        return {
            "model_path": str(output_path),
            "num_reg_images": rec.num_reg_images(),
            "num_points3D": len(rec.points3D),
            "engine": "colmap model_aligner",
        }

    def undistort_images(
        self,
        *,
        model_path: Path,
        image_root: Path,
        output_path: Path,
        spec: dict,
    ) -> dict:
        """Undistort images + adjusted intrinsics via ``image_undistorter``.

        Rewrites every registered image to a distortion-free pinhole
        model and emits the adjusted ``sparse/`` model alongside the
        undistorted images under ``output_path``. ``spec`` carries
        backend-specific ``image_undistorter`` options (``output_type``,
        ``copy_policy``, ``max_image_size``, ``blank_pixels``, the ROI
        bounds, ...) forwarded to the COLMAP CLI verbatim.
        """
        exe = self._require_colmap("image.undistort")
        output_path.mkdir(parents=True, exist_ok=True)
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
        self._append_prefixless_options(
            args,
            spec,
            allowed={
                "output_type",
                "copy_policy",
                "num_patch_match_src_images",
                "blank_pixels",
                "min_scale",
                "max_scale",
                "max_image_size",
                "roi_min_x",
                "roi_min_y",
                "roi_max_x",
                "roi_max_y",
            },
        )
        self._run(args)
        return {
            "output_path": str(output_path),
            "image_path": str(output_path / "images"),
            "model_path": str(output_path / "sparse"),
            "engine": "colmap image_undistorter",
        }

    def build_vocab_tree(
        self,
        *,
        database_path: Path,
        output_path: Path,
        spec: dict,
    ) -> dict:
        """Build a reusable vocabulary-tree index via ``vocab_tree_builder``.

        Trains a visual-word vocabulary tree from the descriptors in a
        COLMAP feature database; the resulting index is what
        ``pairs.vocabtree`` / ``pairs.retrieval`` consume. ``spec``
        carries ``vocab_tree_builder`` options (``num_visual_words``,
        ``num_iterations``, ``num_checks``, ``num_rounds``,
        ``max_num_images``, ``num_threads``) forwarded verbatim.
        """
        exe = self._require_colmap("index.vocab_tree")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        args = [
            exe,
            "vocab_tree_builder",
            "--database_path",
            str(database_path),
            "--vocab_tree_path",
            str(output_path),
        ]
        self._append_prefixless_options(
            args,
            spec,
            allowed={
                "num_visual_words",
                "num_iterations",
                "num_checks",
                "num_threads",
                "num_rounds",
                "max_num_images",
            },
        )
        self._run(args)
        return {
            "vocab_tree_path": str(output_path),
            "database_path": str(database_path),
            "engine": "colmap vocab_tree_builder",
        }

    def configure_rig(
        self,
        *,
        database_path: Path,
        spec: dict,
    ) -> dict:
        """Declare or calibrate a multi-camera rig via ``rig_configurator``.

        COLMAP's ``rig_configurator`` writes rig / frame structure into
        the feature database from a JSON rig config (``rig_config_path``
        in ``spec``), optionally deriving average intrinsics/extrinsics
        from a reference reconstruction (``input_path``). When ``spec``
        carries an inline ``rig_config`` object instead of a path, it is
        serialized to a temp ``.json`` file for the CLI.
        """
        exe = self._require_colmap("rigs.configure")
        normalized = dict(spec or {})
        rig_config_path = normalized.get("rig_config_path")
        inline_config = normalized.get("rig_config")
        cleanup_path: Path | None = None
        if not rig_config_path and inline_config is not None:
            fd, tmp_name = tempfile.mkstemp(suffix=".rig.json")
            cleanup_path = Path(tmp_name)
            with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                json.dump(inline_config, tmp)
            rig_config_path = str(cleanup_path)
        if not rig_config_path:
            raise ValidationError(
                "configure_rig requires spec.rig_config_path or an inline spec.rig_config"
            )
        args = [
            exe,
            "rig_configurator",
            "--database_path",
            str(database_path),
            "--rig_config_path",
            str(rig_config_path),
        ]
        input_path = normalized.get("input_path")
        if input_path:
            args.extend(["--input_path", str(input_path)])
        output_path = normalized.get("output_path")
        if output_path:
            Path(str(output_path)).mkdir(parents=True, exist_ok=True)
            args.extend(["--output_path", str(output_path)])
        try:
            self._run(args)
        finally:
            if cleanup_path is not None:
                cleanup_path.unlink(missing_ok=True)
        result: dict[str, Any] = {
            "database_path": str(database_path),
            "rig_config_path": str(rig_config_path),
            "engine": "colmap rig_configurator",
        }
        if output_path:
            result["model_path"] = str(output_path)
        return result

    def run_colmap_command(
        self,
        command: str,
        *,
        options: dict[str, Any] | None = None,
        positional: list[str | Path] | None = None,
    ) -> dict:
        """Run an arbitrary upstream COLMAP command for demo coverage.

        This is intentionally outside sfmapi's protocol. It lets this package
        expose original COLMAP utilities such as dense stereo, meshing,
        database cleanup, model analysis, and project generation without
        pretending they are standardized sfmapi APIs.
        """
        normalized = command.replace("-", "_").lower()
        if normalized not in COLMAP_COMMANDS:
            raise ValidationError(f"unknown COLMAP command: {command!r}")
        exe = self._require_colmap(f"colmap.{normalized}")
        args = [exe, normalized]
        for item in positional or []:
            args.append(str(item))
        args.extend(self._validated_colmap_option_args(normalized, options or {}))
        result = self._run(args)
        return {
            "command": normalized,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def list_colmap_commands(self) -> list[str]:
        return list(COLMAP_COMMANDS)

    def colmap_command_schema(self, command: str) -> dict[str, Any]:
        normalized = command.replace("-", "_").lower()
        if normalized not in COLMAP_COMMANDS:
            raise ValidationError(f"unknown COLMAP command: {command!r}")
        if normalized in self._command_schema_cache:
            return self._command_schema_cache[normalized]

        exe = self._require_colmap(f"colmap.{normalized}.schema")
        args = [exe, normalized] if normalized == "version" else [exe, normalized, "-h"]
        try:
            result = self._run(args)
        except ValidationError as exc:
            schema = _fallback_colmap_command_schema(normalized, str(exc))
            if schema is None:
                raise
            self._command_schema_cache[normalized] = schema
            return schema
        schema = _parse_colmap_command_help(normalized, result.stdout + result.stderr)
        self._command_schema_cache[normalized] = schema
        return schema

    def list_colmap_command_schemas(self) -> dict[str, dict[str, Any]]:
        return {command: self.colmap_command_schema(command) for command in COLMAP_COMMANDS}

    def _backend_action_command(self, action_id: str) -> str:
        prefix = "colmap."
        if not action_id.startswith(prefix):
            raise ValidationError(f"unknown backend action: {action_id!r}")
        command = action_id.removeprefix(prefix).replace("-", "_").lower()
        if command not in COLMAP_COMMANDS:
            raise ValidationError(f"unknown COLMAP command: {command!r}")
        return command

    def _backend_action_descriptor(
        self,
        command: str,
        *,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        read_only = command in {"help", "version", "model_analyzer", "model_comparer"}
        metadata: dict[str, Any] = {"family": "colmap", "command": command}
        input_schema = None
        if schema is not None:
            metadata["native_schema"] = schema
            metadata["schema_source"] = schema.get("schema_source")
            metadata["option_count"] = schema.get("option_count", len(schema.get("options") or []))
            input_schema = self._backend_action_input_schema(schema)
        return {
            "action_id": f"colmap.{command}",
            "backend": self.name,
            "display_name": f"COLMAP {command}",
            "description": (
                f"Run the upstream COLMAP `{command}` command through the active backend."
            ),
            "category": self._backend_action_category(command),
            "stability": "backend_extension",
            "side_effects": "read" if read_only else "write",
            "long_running": not read_only,
            "supports_progress": False,
            "idempotent": read_only,
            "gpu_required": command
            not in {"help", "version", "model_analyzer", "model_comparer", "database_cleaner"},
            "required_capabilities": [],
            "input_schema": input_schema,
            "output_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "returncode": {"type": "integer"},
                    "stdout": {"type": "string"},
                    "stderr": {"type": "string"},
                },
            },
            "metadata": metadata,
        }

    def _backend_config_descriptor(
        self,
        *,
        config_id: str,
        stage: str,
        capability: str,
        provider: str,
        command: str,
        schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        option_schema = None
        metadata: dict[str, Any] = {
            "family": "colmap",
            "command": command,
        }
        if schema is not None:
            option_schema = self._backend_config_option_schema(schema)
            metadata.update(
                {
                    "schema_source": schema.get("schema_source"),
                    "option_count": schema.get("option_count", len(schema.get("options") or [])),
                }
            )
        return {
            "config_id": config_id,
            "backend": self.name,
            "stage": stage,
            "capability": capability,
            "provider": provider,
            "display_name": f"COLMAP {stage} options",
            "description": (
                f"Backend-specific COLMAP `{command}` options accepted through "
                f"`backend_options` for `{capability}`."
            ),
            "option_schema": option_schema,
            "defaults": {},
            "metadata": metadata,
        }

    def _backend_config_option_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        for option in schema.get("options") or []:
            name = str(option.get("name") or "").strip()
            if not name or name in RUNTIME_MANAGED_COLMAP_OPTIONS:
                continue
            option_schema = dict(option.get("schema") or {"type": "string"})
            description = option.get("description")
            if description and "description" not in option_schema:
                option_schema["description"] = description
            properties[name] = option_schema
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
        }

    def _backend_action_input_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        properties: dict[str, Any] = {
            "positional_args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional positional arguments passed before named options.",
            }
        }
        required: list[str] = []
        for option in schema.get("options") or []:
            name = str(option.get("name") or "").strip()
            if not name:
                continue
            properties[name] = dict(option.get("schema") or {"type": "string"})
            description = option.get("description")
            if description and "description" not in properties[name]:
                properties[name]["description"] = description
            if option.get("required") is True:
                required.append(name)
        out: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
        }
        if required:
            out["required"] = required
        return out

    def _backend_action_category(self, command: str) -> str:
        if "matcher" in command or "verifier" in command:
            return "matching"
        if command in {"feature_extractor", "feature_importer"}:
            return "features"
        if "mapper" in command or command in {"point_triangulator", "bundle_adjuster"}:
            return "mapping"
        if command.startswith("model_") or command in {"image_registrator", "image_deleter"}:
            return "model"
        if command in {"patch_match_stereo", "stereo_fusion", "poisson_mesher", "delaunay_mesher"}:
            return "dense"
        if command.startswith("database_"):
            return "database"
        return "utility"

    def _split_backend_action_inputs(
        self,
        inputs: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        data = dict(inputs or {})
        positional_raw = data.pop("positional_args", data.pop("positional", []))
        if positional_raw is None:
            positional: list[str] = []
        elif isinstance(positional_raw, list):
            positional = [str(item) for item in positional_raw]
        else:
            raise ValidationError("positional_args must be an array of strings")
        if set(data) == {"options"} and isinstance(data.get("options"), dict):
            return dict(data["options"]), positional
        return data, positional

    def _validated_colmap_option_args(
        self,
        command: str,
        options: dict[str, Any],
    ) -> list[str]:
        if not options:
            return []

        schema = self.colmap_command_schema(command)
        if schema.get("available") is False:
            reason = schema.get("unavailable_reason") or "command is unavailable in this build"
            raise ValidationError(f"COLMAP {command} is unavailable: {reason}")

        option_lookup = self._colmap_option_lookup(schema)
        out: list[str] = []
        provided: set[str] = set()
        for raw_key, value in sorted(options.items()):
            if value is None:
                continue
            key = str(raw_key).lstrip("-")
            option = option_lookup.get(self._normalize_colmap_option_key(key))
            if option is None:
                raise ValidationError(f"unknown option for COLMAP {command}: --{key}")
            provided.add(str(option["name"]))
            flag = self._preferred_colmap_option_flag(option)
            if option.get("takes_value") is False:
                if self._option_flag_enabled(value):
                    out.append(flag)
                elif self._option_flag_disabled(value):
                    continue
                else:
                    raise ValidationError(
                        f"option --{option['name']} is a flag and expects a boolean value"
                    )
                continue
            out.extend([flag, self._validated_colmap_option_value(command, option, value)])

        for option in schema.get("options", []):
            if option.get("required") is True and str(option["name"]) not in provided:
                raise ValidationError(
                    f"missing required option for COLMAP {command}: --{option['name']}"
                )
        return out

    def _colmap_option_lookup(self, schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
        lookup: dict[str, dict[str, Any]] = {}
        for option in schema.get("options", []):
            names = [str(option.get("name", ""))]
            names.extend(str(flag).lstrip("-") for flag in option.get("flags", []))
            for name in names:
                if name:
                    lookup[self._normalize_colmap_option_key(name)] = option
        return lookup

    def _normalize_colmap_option_key(self, key: str) -> str:
        return key.strip().lstrip("-").replace("-", "_").lower()

    def _preferred_colmap_option_flag(self, option: dict[str, Any]) -> str:
        for flag in option.get("flags", []):
            if str(flag).startswith("--"):
                return str(flag)
        return f"--{option['name']}"

    def _validated_colmap_option_value(
        self,
        command: str,
        option: dict[str, Any],
        value: Any,
    ) -> str:
        option_name = str(option["name"])
        option_type = str(option.get("type") or option.get("schema", {}).get("type") or "string")
        choices = [str(choice) for choice in option.get("choices") or []]
        if not choices:
            choices = [str(choice) for choice in option.get("schema", {}).get("enum") or []]

        if option_type == "integer":
            if isinstance(value, bool):
                raise ValidationError(
                    f"option --{option_name} for COLMAP {command} expects integer"
                )
            try:
                parsed = int(value)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"option --{option_name} for COLMAP {command} expects integer: {value!r}"
                ) from exc
            return str(parsed)

        if option_type == "number":
            if isinstance(value, bool):
                raise ValidationError(f"option --{option_name} for COLMAP {command} expects number")
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"option --{option_name} for COLMAP {command} expects number: {value!r}"
                ) from exc
            return str(parsed)

        if option_type == "boolean":
            if isinstance(value, bool):
                return "1" if value else "0"
            lowered = str(value).strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return "1"
            if lowered in {"0", "false", "no", "off"}:
                return "0"
            raise ValidationError(f"option --{option_name} for COLMAP {command} expects boolean")

        if isinstance(value, (list, tuple)):
            text = ",".join(str(item) for item in value)
        else:
            text = str(value)
        if choices and text not in choices:
            raise ValidationError(
                f"option --{option_name} for COLMAP {command} must be one of "
                f"{', '.join(choices)}: {text!r}"
            )
        return text

    def _option_flag_enabled(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _option_flag_disabled(self, value: Any) -> bool:
        if isinstance(value, bool):
            return not value
        return str(value).strip().lower() in {"0", "false", "no", "off"}

    def read_reconstruction(self, path: Path) -> Reconstruction:
        model_path = Path(path)
        if (model_path / "cameras.txt").exists():
            return read_colmap_text_model(model_path)
        with tempfile.TemporaryDirectory(prefix="sfmapi-colmap-model-") as tmp:
            text_dir = Path(tmp)
            self._convert_model(model_path, text_dir, "TXT")
            return read_colmap_text_model(text_dir)

    def runtime_versions(self) -> dict[str, str]:
        exe = self._find_colmap()
        versions = {
            "backend": self.version,
            "colmap_executable": str(exe) if exe else "missing",
            "colmap_source_sha": self._colmap_source_sha(),
        }
        if exe is not None:
            versions["colmap_help_header"] = self._colmap_help_header(exe)
            try:
                stat = exe.stat()
            except OSError:
                pass
            else:
                versions["colmap_executable_size"] = str(stat.st_size)
                versions["colmap_executable_mtime_ns"] = str(stat.st_mtime_ns)
        return versions

    def _require_colmap(self, capability: str) -> str:
        exe = self._find_colmap()
        if exe is None:
            raise CapabilityUnavailableError(
                capability=capability,
                reason=(
                    "COLMAP executable not found. Set SFMAPI_COLMAP_EXECUTABLE, "
                    "put colmap on PATH, or build third_party/colmap."
                ),
            )
        return str(exe)

    def _find_colmap(self) -> Path | None:
        if self._executable_override and self._executable_override.exists():
            return self._executable_override

        env = os.environ.get("SFMAPI_COLMAP_EXECUTABLE")
        if env and Path(env).exists():
            return Path(env)

        names = ["colmap.exe", "colmap.bat", "colmap"] if os.name == "nt" else ["colmap"]
        project_root = REPO_ROOT.parent
        candidates: list[Path] = []
        for name in names:
            candidates.extend(
                [
                    REPO_ROOT / "third_party" / "colmap" / "install" / "bin" / name,
                    project_root / "colmap-install-cuda-cudss" / "bin" / name,
                    REPO_ROOT
                    / "third_party"
                    / "colmap"
                    / "build-cuda-cudss"
                    / "src"
                    / "colmap"
                    / "exe"
                    / name,
                    REPO_ROOT
                    / "third_party"
                    / "colmap"
                    / "build"
                    / "src"
                    / "colmap"
                    / "exe"
                    / name,
                    REPO_ROOT
                    / "third_party"
                    / "colmap"
                    / "build"
                    / "src"
                    / "colmap"
                    / "exe"
                    / "Release"
                    / name,
                    REPO_ROOT / "third_party" / "colmap" / "build" / "src" / "exe" / name,
                    REPO_ROOT / "third_party" / "colmap" / "build" / "bin" / name,
                    REPO_ROOT / "third_party" / "colmap" / "build" / name,
                ]
            )
        candidates.extend(_cached_colmap_candidates())
        for candidate in candidates:
            if candidate.exists():
                return candidate

        found = shutil.which("colmap")
        if found:
            return Path(found)
        return None

    def _convert_model(self, input_path: Path, output_path: Path, output_type: str) -> None:
        exe = self._require_colmap(f"export.{output_type.lower()}")
        self._run(
            [
                exe,
                "model_converter",
                "--input_path",
                str(input_path),
                "--output_path",
                str(output_path),
                "--output_type",
                output_type,
            ]
        )

    def _run(
        self,
        args: list[str],
        *,
        progress: Any | None = None,
        progress_phase: str | None = None,
        progress_total: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        executable = Path(args[0]) if args else None
        if progress is not None and progress_phase is not None:
            return self._run_with_progress(
                args,
                progress=progress,
                progress_phase=progress_phase,
                progress_total=progress_total,
                executable=executable,
            )
        try:
            return subprocess.run(
                args,
                check=True,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
                env=colmap_runtime_env(executable),
            )
        except subprocess.CalledProcessError as exc:
            command = " ".join(args[:2])
            detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            raise ValidationError(f"{command} failed: {detail}") from exc

    def _run_stage(
        self,
        args: list[str],
        *,
        progress: Any | None,
        progress_phase: str,
        progress_total: int | None,
    ) -> subprocess.CompletedProcess[str]:
        if progress is None:
            return self._run(args)
        return self._run(
            args,
            progress=progress,
            progress_phase=progress_phase,
            progress_total=progress_total,
        )

    def _run_with_progress(
        self,
        args: list[str],
        *,
        progress: Any,
        progress_phase: str,
        progress_total: int | None,
        executable: Path | None,
    ) -> subprocess.CompletedProcess[str]:
        normalized_args = [str(arg) for arg in args]
        process = subprocess.Popen(
            normalized_args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=colmap_runtime_env(executable),
            bufsize=1,
        )
        output_lines: list[str] = []
        if process.stdout is not None:
            for line in process.stdout:
                output_lines.append(line)
                self._progress_from_line(progress, progress_phase, line, progress_total)
        returncode = process.wait()
        stdout = "".join(output_lines)
        result = subprocess.CompletedProcess(normalized_args, returncode, stdout, "")
        if result.returncode != 0:
            command = " ".join(normalized_args[:2])
            detail = result.stdout.strip() or f"exit {result.returncode}"
            raise ValidationError(f"{command} failed: {detail}") from None
        return result

    def _progress_from_line(
        self,
        progress: Any,
        phase: str,
        line: str,
        total_hint: int | None,
    ) -> None:
        match = _BRACKET_PROGRESS_RE.search(line)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            self._progress(progress, phase, current=current, total=total)
            return
        percent_match = _PERCENT_PROGRESS_RE.search(line)
        if percent_match is None:
            return
        percent = max(0, min(100, int(percent_match.group(1))))
        total = total_hint or 100
        current = round((percent / 100.0) * total)
        self._progress(progress, phase, current=current, total=total)

    def _progress(
        self,
        progress: Any | None,
        phase: str,
        *,
        current: int,
        total: int | None,
    ) -> None:
        if progress is None:
            return
        try:
            progress.phase_progress(phase, current=max(0, current), total=total)
        except Exception:
            return

    def _database_image_count(self, database_path: Path) -> int | None:
        try:
            with sqlite3.connect(database_path) as conn:
                row = conn.execute("select count(*) from images").fetchone()
        except sqlite3.Error:
            return None
        return int(row[0]) if row is not None else None

    def _database_match_pair_count(self, database_path: Path) -> int | None:
        try:
            with sqlite3.connect(database_path) as conn:
                row = conn.execute("select count(*) from matches where rows > 0").fetchone()
        except sqlite3.Error:
            return None
        return int(row[0]) if row is not None else None

    def _match_progress_total(
        self,
        database_path: Path,
        mode: str,
        options: dict[str, Any],
    ) -> int | None:
        num_images = self._database_image_count(database_path)
        if num_images is None:
            return None
        if mode == "exhaustive":
            return max(0, num_images * (num_images - 1) // 2)
        if mode == "sequential":
            overlap = int(options.get("overlap") or 10)
            return sum(min(overlap, num_images - 1 - index) for index in range(num_images - 1))
        if mode == "explicit":
            pairs_path = options.get("match_list_path") or options.get("pairs_path")
            if pairs_path:
                with Path(str(pairs_path)).open("r", encoding="utf-8") as fh:
                    return sum(1 for line in fh if line.strip() and not line.startswith("#"))
        return None

    def _mapping_progress_phase(self, kind: str) -> str:
        return {
            "global": "global_positioning",
            "glomap": "global_positioning",
            "incremental": "incremental_register",
            "hierarchical": "hierarchical_cluster",
            "spherical": "spherical",
        }.get(kind, "incremental_register")

    def _append_options(self, args: list[str], prefix: str, options: dict[str, Any]) -> None:
        mapped: dict[str, tuple[int, Any]] = {}
        for source_rank, source in enumerate((options, options.get("backend_options") or {})):
            if not isinstance(source, dict):
                continue
            for key, value in sorted(source.items()):
                if value is None or isinstance(value, (dict, list, tuple)):
                    continue
                option_key = str(key)
                if option_key in {
                    "type",
                    "strategy",
                    "mode",
                    "matcher",
                    "backend_options",
                    "portable",
                    "legacy_options",
                }:
                    continue
                explicit = "." in option_key
                colmap_key = option_key if explicit else f"{prefix}.{option_key}"
                self._set_colmap_option(
                    mapped,
                    colmap_key,
                    value,
                    priority=self._colmap_option_priority(source_rank, explicit=explicit),
                )
        self._append_mapped_options(args, mapped)

    def _append_feature_options(self, args: list[str], options: dict[str, Any]) -> None:
        mapped: dict[str, tuple[int, Any]] = {}
        sources = (
            options,
            options.get("sift") or {},
            options.get("extractor_options") or {},
            options.get("backend_options") or {},
        )
        feature_keys = {"use_gpu", "gpu_index", "num_threads"}
        image_reader_keys = {
            "camera_model",
            "camera_params",
            "camera_mode",
            "default_focal_length_factor",
            "existing_camera_id",
            "single_camera",
            "single_camera_per_folder",
            "single_camera_per_image",
        }
        for source_rank, source in enumerate(sources):
            if not isinstance(source, dict):
                continue
            for key, value in sorted(source.items()):
                if value is None or isinstance(value, (dict, list, tuple)):
                    continue
                option_key = str(key)
                if option_key in {
                    "version",
                    "type",
                    "provider",
                    "seed",
                    "sift",
                    "extractor_options",
                    "backend_options",
                    "portable",
                    "legacy_options",
                }:
                    continue
                explicit = "." in option_key
                if explicit:
                    colmap_key = option_key
                elif option_key in feature_keys:
                    colmap_key = f"FeatureExtraction.{option_key}"
                elif option_key in image_reader_keys:
                    colmap_key = f"ImageReader.{option_key}"
                else:
                    colmap_key = f"SiftExtraction.{option_key}"
                self._set_colmap_option(
                    mapped,
                    colmap_key,
                    value,
                    priority=self._colmap_option_priority(source_rank, explicit=explicit),
                )
        self._append_mapped_options(args, mapped)

    def _append_match_options(self, args: list[str], command: str, options: dict[str, Any]) -> None:
        mapped: dict[str, tuple[int, Any]] = {}
        root_backend_options = (
            options.get("backend_options")
            if isinstance(options.get("backend_options"), dict)
            else {}
        )
        pairs_options = options.get("pairs") if isinstance(options.get("pairs"), dict) else {}
        matcher_options = options.get("matcher") if isinstance(options.get("matcher"), dict) else {}
        sources = (
            options,
            root_backend_options.get("pairs") if isinstance(root_backend_options, dict) else {},
            root_backend_options.get("matcher") if isinstance(root_backend_options, dict) else {},
            pairs_options,
            pairs_options.get("backend_options") if isinstance(pairs_options, dict) else {},
            matcher_options,
            matcher_options.get("backend_options") if isinstance(matcher_options, dict) else {},
            options.get("matcher_options") or {},
        )
        feature_matching_keys = {
            "guided_matching",
            "gpu_index",
            "max_num_matches",
            "num_threads",
            "rig_verification",
            "use_gpu",
        }
        sift_matching_keys = {
            "cross_check",
            "cpu_brute_force_matcher",
            "max_distance",
            "max_ratio",
        }
        command_prefix = self._matcher_option_prefix(command)
        command_keys = {
            "exhaustive_matcher": {"block_size"},
            "sequential_matcher": {"loop_detection", "loop_detection_num_images", "overlap"},
            "spatial_matcher": {"ignore_z", "max_distance", "max_num_neighbors"},
            "vocab_tree_matcher": {
                "match_list_path",
                "num_images",
                "num_nearest_neighbors",
                "vocab_tree_path",
            },
            "matches_importer": {"match_list_path", "match_type"},
            "transitive_matcher": {"batch_size", "num_iterations"},
        }.get(command, set())

        for source_rank, source in enumerate(sources):
            if not isinstance(source, dict):
                continue
            for key, value in sorted(source.items()):
                if value is None or isinstance(value, (dict, list, tuple)):
                    continue
                option_key = str(key)
                if option_key in {
                    "version",
                    "strategy",
                    "type",
                    "mode",
                    "matcher",
                    "pairs",
                    "matcher_options",
                    "backend_options",
                    "portable",
                    "legacy_options",
                    "provider",
                    "pairs_provider",
                    "matcher_provider",
                    "image_pairs",
                    "pairs_blob_sha",
                    "pairs_blob_format",
                    "pairs_path",
                }:
                    continue
                explicit = "." in option_key
                if explicit:
                    colmap_key = option_key
                elif option_key in feature_matching_keys:
                    colmap_key = f"FeatureMatching.{option_key}"
                elif option_key in sift_matching_keys:
                    colmap_key = f"SiftMatching.{option_key}"
                elif command == "matches_importer" and option_key in command_keys:
                    colmap_key = option_key
                elif option_key in command_keys:
                    colmap_key = f"{command_prefix}.{option_key}"
                else:
                    continue
                self._set_colmap_option(
                    mapped,
                    colmap_key,
                    value,
                    priority=self._colmap_option_priority(source_rank, explicit=explicit),
                )
        self._append_mapped_options(args, mapped)

    def _colmap_option_priority(self, source_rank: int, *, explicit: bool) -> int:
        return source_rank * 2 + int(explicit)

    def _set_colmap_option(
        self,
        mapped: dict[str, tuple[int, Any]],
        colmap_key: str,
        value: Any,
        *,
        priority: int,
    ) -> None:
        current = mapped.get(colmap_key)
        if current is None or priority >= current[0]:
            mapped[colmap_key] = (priority, value)

    def _append_mapped_options(
        self,
        args: list[str],
        mapped: dict[str, tuple[int, Any]],
    ) -> None:
        for colmap_key, (_priority, value) in sorted(mapped.items()):
            args.extend([f"--{colmap_key}", self._stringify_option(value)])

    def _explicit_match_options(
        self, options: dict[str, Any]
    ) -> tuple[dict[str, Any], Path | None]:
        normalized = dict(options or {})
        pairs_spec = normalized.get("pairs") if isinstance(normalized.get("pairs"), dict) else {}
        match_list_path = (
            normalized.get("match_list_path")
            or normalized.get("pairs_path")
            or pairs_spec.get("match_list_path")
            or pairs_spec.get("pairs_path")
        )
        if match_list_path:
            normalized["match_list_path"] = str(match_list_path)
            normalized.setdefault("match_type", pairs_spec.get("match_type") or "pairs")
            return normalized, None

        image_pairs = normalized.get("image_pairs") or pairs_spec.get("image_pairs")
        if not image_pairs:
            raise ValidationError("explicit pair matching requires match_list_path or image_pairs")

        fd, tmp_name = tempfile.mkstemp(suffix=".pairs.txt")
        cleanup_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                for pair in image_pairs:
                    image_name1, image_name2 = self._explicit_pair_names(pair)
                    tmp.write(f"{image_name1} {image_name2}\n")
        except Exception:
            cleanup_path.unlink(missing_ok=True)
            raise
        normalized["match_list_path"] = str(cleanup_path)
        normalized.setdefault("match_type", "pairs")
        return normalized, cleanup_path

    def _explicit_pair_names(self, pair: Any) -> tuple[str, str]:
        if isinstance(pair, dict):
            image_name1 = str(pair.get("image_name1") or "")
            image_name2 = str(pair.get("image_name2") or "")
        elif isinstance(pair, (list, tuple)) and len(pair) >= 2:
            image_name1 = str(pair[0])
            image_name2 = str(pair[1])
        else:
            raise ValidationError("explicit image pairs must be objects or 2-item arrays")
        if not image_name1 or not image_name2 or image_name1 == image_name2:
            raise ValidationError("explicit image pairs require two different image names")
        return image_name1, image_name2

    def _append_verify_options(self, args: list[str], options: dict[str, Any]) -> None:
        flattened: dict[str, Any] = {}
        for source in (options, options.get("backend_options") or {}):
            if isinstance(source, dict):
                flattened.update(source)
        for key, value in sorted(flattened.items()):
            if value is None or isinstance(value, (dict, list, tuple)):
                continue
            option_key = str(key)
            if option_key in {
                "version",
                "provider",
                "use_gpu",
                "backend_options",
                "portable",
                "legacy_options",
            }:
                continue
            colmap_key = option_key if "." in option_key else f"TwoViewGeometry.{option_key}"
            args.extend([f"--{colmap_key}", self._stringify_option(value)])

    def _append_mapper_options(
        self,
        args: list[str],
        spec: dict[str, Any],
        command: str = "mapper",
    ) -> None:
        source = spec.get("mapper") or spec.get("options") or spec
        if not isinstance(source, dict):
            return
        flattened = dict(source)
        backend_options = source.get("backend_options")
        if isinstance(backend_options, dict):
            flattened.update(backend_options)
        skip_keys = {
            "ba_global_use_pba",
            "backend",
            "backend_options",
            "formulation",
            "init_image_pair",
            "kind",
            "max_runtime_seconds",
            "portable",
            "legacy_options",
            "provider",
            "snapshot_frames_freq",
            "use_incremental_quality_fallback",
            "version",
        }
        for key, value in sorted(flattened.items()):
            if value is None or isinstance(value, (dict, list, tuple)):
                continue
            option_key = str(key)
            if option_key in skip_keys:
                continue
            if "." in option_key:
                colmap_key = option_key
            elif option_key == "seed":
                colmap_key = (
                    "default_random_seed" if command == "global_mapper" else "Mapper.random_seed"
                )
            else:
                prefix = "GlobalMapper" if command == "global_mapper" else "Mapper"
                colmap_key = f"{prefix}.{option_key}"
            args.extend([f"--{colmap_key}", self._stringify_option(value)])

    def _append_prefixless_options(
        self,
        args: list[str],
        spec: dict[str, Any],
        *,
        allowed: set[str],
    ) -> None:
        """Append scalar ``spec`` options that name COLMAP CLI flags directly.

        Used by the standalone-utility wrappers (``model_aligner``,
        ``image_undistorter``, ``vocab_tree_builder``) whose options are
        plain top-level CLI flags with no ``Group.option`` prefix. Only
        keys in ``allowed`` (or already dotted/prefixed keys) are
        forwarded; everything else — discriminators, nested dicts,
        runtime-managed paths — is dropped. ``backend_options`` is
        merged in with higher priority, mirroring the stage wrappers.
        """
        flattened: dict[str, Any] = {}
        for source in (spec, spec.get("options") or {}, spec.get("backend_options") or {}):
            if isinstance(source, dict):
                for key, value in source.items():
                    flattened[str(key)] = value
        for key, value in sorted(flattened.items()):
            if value is None or isinstance(value, (dict, list, tuple)):
                continue
            # Dotted keys are passed through verbatim so callers can
            # still reach ``Group.option`` flags this wrapper has not
            # enumerated; plain keys must be in the allow-list.
            if "." in key or key in allowed:
                args.extend([f"--{key}", self._stringify_option(value)])

    def _append_rotation_averager_options(self, args: list[str], options: dict[str, Any]) -> None:
        """Append ``rotation_averager`` options from a mapping spec.

        COLMAP's ``rotation_averager`` takes plain top-level flags. Only
        the known set is forwarded; dotted keys are passed through so
        callers can still reach options this wrapper has not enumerated.
        """
        self._append_prefixless_options(
            args,
            options if isinstance(options, dict) else {},
            allowed={
                "max_num_iterations",
                "use_weights",
                "skip_initialization",
                "num_threads",
            },
        )

    def _write_pose_priors(self, database_path: Path, pose_priors: dict[str, Any]) -> int:
        """Materialize sfmapi pose priors into COLMAP's ``pose_priors`` table.

        sfmapi keeps per-image priors in its own store; COLMAP's
        ``pose_prior_mapper`` reads them from the feature database. This
        bridges the two: for every prior whose image name resolves to a
        row in ``images``, write a ``pose_priors`` row with the position
        (GPS lon/lat/alt when present, else the prior translation) and,
        when available, the 3x3 position block of the 6x6 covariance.
        Returns the number of rows written.
        """
        if not pose_priors:
            return 0
        written = 0
        with sqlite3.connect(database_path) as conn:
            image_ids = {
                str(name): int(image_id)
                for image_id, name in conn.execute("select image_id, name from images").fetchall()
            }
            conn.execute(
                "create table if not exists pose_priors ("
                "image_id integer primary key, position blob, "
                "coordinate_system integer, position_covariance blob)"
            )
            for image_name, prior in pose_priors.items():
                image_id = image_ids.get(str(image_name))
                if image_id is None or not isinstance(prior, dict):
                    continue
                position, coordinate_system = self._pose_prior_position(prior)
                if position is None:
                    continue
                covariance = self._pose_prior_position_covariance(prior)
                conn.execute(
                    "insert or replace into pose_priors "
                    "(image_id, position, coordinate_system, position_covariance) "
                    "values (?, ?, ?, ?)",
                    (
                        image_id,
                        struct.pack("<3d", *position),
                        coordinate_system,
                        struct.pack("<9d", *covariance) if covariance else None,
                    ),
                )
                written += 1
            conn.commit()
        return written

    def _pose_prior_position(
        self, prior: dict[str, Any]
    ) -> tuple[tuple[float, float, float] | None, int]:
        """Return ``((x, y, z), coordinate_system)`` for a sfmapi PosePrior.

        Prefers GPS ``(lon, lat, alt)`` (COLMAP coordinate system ``1``);
        falls back to the ``cam_from_world`` translation (``-1``,
        unknown/cartesian). Returns ``(None, ...)`` when neither is
        usable.
        """
        gps = prior.get("gps")
        if isinstance(gps, dict):
            lon = gps.get("longitude") or gps.get("lon")
            lat = gps.get("latitude") or gps.get("lat")
            alt = gps.get("altitude") or gps.get("alt") or 0.0
            if lon is not None and lat is not None:
                return (float(lon), float(lat), float(alt)), 1
        cam_from_world = prior.get("cam_from_world")
        if isinstance(cam_from_world, dict):
            translation = cam_from_world.get("translation") or cam_from_world.get("t")
            if isinstance(translation, (list, tuple)) and len(translation) >= 3:
                return (
                    float(translation[0]),
                    float(translation[1]),
                    float(translation[2]),
                ), -1
        return None, -1

    def _pose_prior_position_covariance(self, prior: dict[str, Any]) -> list[float] | None:
        """Extract the 3x3 position covariance from a sfmapi PosePrior.

        sfmapi stores a 36-float row-major 6x6 covariance ordered
        ``(rx, ry, rz, tx, ty, tz)``; COLMAP's ``pose_priors`` table
        wants the 3x3 *position* block. Slice the lower-right ``tx,ty,tz``
        sub-matrix out of the 6x6. Returns ``None`` when no covariance
        is present.
        """
        covariance = prior.get("covariance")
        if not isinstance(covariance, (list, tuple)) or len(covariance) != 36:
            return None
        block: list[float] = []
        for row in (3, 4, 5):
            for col in (3, 4, 5):
                block.append(float(covariance[row * 6 + col]))
        return block

    def _stringify_option(self, value: Any) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        return str(value)

    def _matcher_option_prefix(self, command: str) -> str:
        if command == "exhaustive_matcher":
            return "ExhaustiveMatching"
        if command == "sequential_matcher":
            return "SequentialMatching"
        if command == "spatial_matcher":
            return "SpatialMatching"
        if command == "vocab_tree_matcher":
            return "VocabTreeMatching"
        if command == "transitive_matcher":
            return "TransitiveMatching"
        return "Matching"

    def _match_capability(self, mode: str) -> str:
        if mode == "from_poses":
            return "pairs.from_poses"
        if mode in {"exhaustive", "sequential", "spatial", "vocabtree", "vocab_tree", "explicit"}:
            return f"pairs.{mode.replace('_', '')}"
        return "backend.actions"

    def _decode_keypoints(self, *, rows: int, cols: int, blob: bytes) -> list[list[float]]:
        if rows <= 0 or cols <= 0 or not blob:
            return []
        values = struct.unpack(f"<{len(blob) // 4}f", blob)
        out: list[list[float]] = []
        for row in range(rows):
            start = row * cols
            vals = values[start : start + cols]
            x = float(vals[0]) if len(vals) > 0 else 0.0
            y = float(vals[1]) if len(vals) > 1 else 0.0
            scale = float(vals[2]) if len(vals) > 2 else 1.0
            angle = float(vals[3]) if len(vals) > 3 else 0.0
            out.append([x, y, scale, angle])
        return out

    def _decode_descriptor_bytes(self, *, rows: int, cols: int, blob: bytes) -> bytes:
        if rows <= 0 or cols <= 0 or not blob:
            return b""
        expected = rows * cols
        values = blob[:expected]
        # COLMAP stores SIFT descriptors as uint8. sfmapi's oneshot helper
        # expects raw float32 bytes, so widen losslessly for transport.
        return struct.pack(f"<{len(values)}f", *(float(v) for v in values))

    def _decode_uint_matrix(
        self, rows: int, cols: int, blob: bytes | memoryview | None
    ) -> list[tuple[int, int]]:
        if rows <= 0 or cols < 2 or not blob:
            return []
        raw = bytes(blob)
        values = struct.unpack(f"<{len(raw) // 4}I", raw)
        pairs: list[tuple[int, int]] = []
        for row in range(rows):
            start = row * cols
            if start + 1 >= len(values):
                break
            pairs.append((int(values[start]), int(values[start + 1])))
        return pairs

    def _decode_matrix_3x3(self, blob: bytes | memoryview | None) -> list[list[float]] | None:
        values = self._decode_float64_vector(blob)
        if values is None or len(values) != 9:
            return None
        return [values[0:3], values[3:6], values[6:9]]

    def _decode_float64_vector(self, blob: bytes | memoryview | None) -> list[float] | None:
        if not blob:
            return None
        raw = bytes(blob)
        if len(raw) % 8 != 0:
            return None
        return [float(v) for v in struct.unpack(f"<{len(raw) // 8}d", raw)]

    def _pair_id_to_image_ids(self, pair_id: int) -> tuple[int, int]:
        image_id2 = pair_id % COLMAP_PAIR_ID_BASE
        image_id1 = (pair_id - image_id2) // COLMAP_PAIR_ID_BASE
        return int(image_id1), int(image_id2)

    def _sim3_to_transform_text(self, sim3: dict) -> str:
        scale = float(sim3.get("scale", 1.0))
        rotation = sim3.get("rotation") or {}
        translation = sim3.get("translation") or (0.0, 0.0, 0.0)
        if hasattr(rotation, "model_dump"):
            rotation = rotation.model_dump()
        if hasattr(translation, "model_dump"):
            translation = translation.model_dump()
        w = float(rotation.get("w", 1.0))
        x = float(rotation.get("x", 0.0))
        y = float(rotation.get("y", 0.0))
        z = float(rotation.get("z", 0.0))
        tx, ty, tz = (float(v) for v in translation)
        matrix = self._quat_wxyz_to_matrix(w, x, y, z)
        rows = [
            [scale * matrix[0][0], scale * matrix[0][1], scale * matrix[0][2], tx],
            [scale * matrix[1][0], scale * matrix[1][1], scale * matrix[1][2], ty],
            [scale * matrix[2][0], scale * matrix[2][1], scale * matrix[2][2], tz],
            [0.0, 0.0, 0.0, 1.0],
        ]
        return "\n".join(" ".join(f"{value:.17g}" for value in row) for row in rows) + "\n"

    def _quat_wxyz_to_matrix(self, w: float, x: float, y: float, z: float) -> list[list[float]]:
        norm = (w * w + x * x + y * y + z * z) ** 0.5
        if norm == 0:
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        return [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]

    def _colmap_source_sha(self) -> str:
        submodule = REPO_ROOT / "third_party" / "colmap"
        if not (submodule / ".git").exists() and not (submodule / ".git").is_file():
            return "missing"
        try:
            result = subprocess.run(
                ["git", "-C", str(submodule), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
            )
        except Exception:
            return "unknown"
        return result.stdout.strip() or "unknown"

    def _colmap_help_header(self, exe: Path) -> str:
        try:
            result = subprocess.run(
                [str(exe), "-h"],
                check=False,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
                timeout=10,
                env=colmap_runtime_env(exe),
            )
        except Exception:
            return "unknown"
        for line in (result.stdout + result.stderr).splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:160]
        return "unknown"


__all__ = ["ColmapCliBackend"]
