from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
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
DEFAULT_INSTANTSFM_ROOT = REPO_ROOT / "third_party" / "instantsfm"
# The SciPy-backed ``sksparse.cholmod`` shim ships as plugin-private package
# data -- deliberately NOT as an installable top-level ``sksparse`` package,
# which would shadow a real scikit-sparse everywhere in the venv. InstantSfM
# worker subprocesses reach it through PYTHONPATH injection only (see
# :func:`instantsfm_pythonpath`).
SKSPARSE_SHIM_DIR = Path(__file__).resolve().parent / "_sksparse_shim"
FEATURE_HANDLERS = (
    "colmap",
    "dedode",
    "disk+lightglue",
    "superpoint+lightglue",
    "sift",
)


@dataclass(frozen=True)
class InstantSfMCommand:
    action_id: str
    display_name: str
    category: str
    module: str
    description: str
    gpu_required: bool = True


INSTANTSFM_COMMANDS: tuple[InstantSfMCommand, ...] = (
    InstantSfMCommand(
        "instantsfm.extractFeatures",
        "InstantSfM feature extraction",
        "features",
        "instantsfm.scripts.feat",
        "Extract and match features into the InstantSfM/COLMAP database.",
    ),
    InstantSfMCommand(
        "instantsfm.runGlobalSfm",
        "InstantSfM global SfM",
        "mapping",
        "instantsfm.scripts.sfm",
        "Run InstantSfM global mapping and write a sparse reconstruction.",
    ),
    InstantSfMCommand(
        "instantsfm.trainGaussianSplatting",
        "InstantSfM 3DGS training",
        # 3DGS training produces a radiance field, not a dense MVS mesh /
        # point cloud. The category label is descriptive only -- 3DGS
        # training is still exposed as a backend action, not a portable
        # sfmapi capability.
        "radiance_field",
        "instantsfm.scripts.gs",
        "Train the optional Gaussian Splatting viewer output.",
    ),
    InstantSfMCommand(
        "instantsfm.visualizeReconstruction",
        "InstantSfM reconstruction visualizer",
        "visualization",
        "instantsfm.scripts.vis_recon",
        "Open the InstantSfM offline reconstruction visualizer.",
        gpu_required=False,
    ),
)
_COMMAND_BY_ACTION = {command.action_id: command for command in INSTANTSFM_COMMANDS}
_SCRIPT_MODULES = {command.module for command in INSTANTSFM_COMMANDS}


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


def resolve_instantsfm_root(value: str | Path | None) -> Path | None:
    raw = value or os.environ.get("SFMAPI_INSTANTSFM_ROOT")
    cache_root = _plugin_cache_root("instantsfm")
    candidates = (
        [_expand_path(raw)]
        if raw
        else [
            DEFAULT_INSTANTSFM_ROOT,
            cache_root / "plugin_source" / "third_party" / "instantsfm",
            # Legacy provisioner cache path from before the wrapper repo
            # owned the upstream checkout as a submodule.
            cache_root / "source",
        ]
    )
    for path in candidates:
        if (path / "pyproject.toml").exists() and (path / "instantsfm").is_dir():
            return path.resolve()
    return None


def instantsfm_pythonpath(root: str | Path, existing: str | None = None) -> str:
    """Build the ``PYTHONPATH`` InstantSfM worker processes run with.

    Prepends the upstream checkout (so ``instantsfm.*`` resolves) and the
    plugin-private shim directory (so the vendored engine's
    ``import sksparse.cholmod`` resolves inside the worker subprocess without
    the shim ever shadowing a real scikit-sparse install in site-packages).
    Every PYTHONPATH injection for InstantSfM must go through this helper.
    """
    prepend = [str(root), str(SKSPARSE_SHIM_DIR)]
    parts = [part for part in (existing or "").split(os.pathsep) if part]
    return os.pathsep.join([*prepend, *[part for part in parts if part not in prepend]])


def configure_instantsfm_environment(
    root: str | Path | None = None,
    *,
    python_executable: str | Path | None = None,
    validate: bool = False,
) -> Path | None:
    resolved_root = resolve_instantsfm_root(root)
    if resolved_root is None:
        if validate:
            raise ValueError(
                "InstantSfM checkout not found. Set SFMAPI_INSTANTSFM_ROOT or pass "
                "--instantsfm-root to sfmapi-instantsfm-api."
            )
        return None

    os.environ["SFMAPI_INSTANTSFM_ROOT"] = str(resolved_root)
    python = Path(python_executable or os.environ.get("SFMAPI_INSTANTSFM_PYTHON") or sys.executable)
    os.environ["SFMAPI_INSTANTSFM_PYTHON"] = str(python)
    os.environ["PYTHONPATH"] = instantsfm_pythonpath(resolved_root, os.environ.get("PYTHONPATH"))
    return resolved_root


class InstantSfMBackend:
    name = "instantsfm"
    version = "0.0.1"
    vendor = "InstantSfM"

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        python_executable: str | Path | None = None,
    ) -> None:
        self._root_override = _expand_path(root).resolve() if root else None
        self._python_executable = Path(
            python_executable or os.environ.get("SFMAPI_INSTANTSFM_PYTHON") or sys.executable
        )

    def capabilities(self) -> set[str]:
        """Portable sfmapi capabilities backed by real wrapper methods.

        InstantSfM is a *global* SfM engine. ``run_mapping`` wraps
        ``instantsfm.scripts.sfm`` (which emits standard COLMAP
        ``cameras.bin`` / ``images.bin`` / ``points3D.bin``) behind a
        path-staging adapter, so the portable ``map.global`` stage is
        advertised. Feature extraction stays action-only: InstantSfM's
        ``scripts.feat`` fuses extraction + matching into one
        whole-project ``GenerateDatabase`` call with no separable
        extract / pairs / match stages. The set is empty until the
        InstantSfM checkout is resolvable -- a capability the deployment
        cannot actually run must not be advertised.
        """
        if self._find_root() is None:
            return set()
        return {"map.global"}

    def list_backend_config_schemas(self, *, include_schemas: bool = True) -> list[dict[str, Any]]:
        """Portable option schema for the global mapping stage.

        Exposes the InstantSfM-specific knobs that ``run_mapping`` forwards to
        ``instantsfm.scripts.sfm`` so clients can discover and validate them
        through the standard ``backend_options`` envelope (previously the
        manifest advertised no config schemas for ``map.global``).
        """
        if self._find_root() is None:
            return []
        option_schema = None
        if include_schemas:
            option_schema = {
                "type": "object",
                # The 5 knobs below are the complete set run_mapping consumes;
                # closing the schema lets sfmapi reject misspelled backend_options
                # (and satisfies backend_config_contract_violations, which
                # mandates additionalProperties: false).
                "additionalProperties": False,
                "properties": {
                    "export_txt": {
                        "type": "boolean",
                        "default": False,
                        "description": "Also export the model as COLMAP text "
                        "(cameras.txt/images.txt/points3D.txt); export_text is an alias.",
                    },
                    "disable_depths": {
                        "type": "boolean",
                        "default": False,
                        "description": "Disable monocular-depth priors during global SfM.",
                    },
                    "disable_semantics": {
                        "type": "boolean",
                        "default": False,
                        "description": "Disable semantic filtering during global SfM.",
                    },
                    "manual_config_name": {
                        "type": "string",
                        "description": "Name of a bundled InstantSfM config preset to use.",
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Hard wall-clock timeout for the mapping subprocess.",
                    },
                },
            }
        return [
            {
                "config_id": "instantsfm.mapping.global",
                "backend": self.name,
                "stage": "mapping",
                "capability": "map.global",
                "provider": "instantsfm",
                "display_name": "InstantSfM global mapping options",
                "description": "InstantSfM-specific options for the portable map.global stage, "
                "forwarded through backend_options.",
                "option_schema": option_schema,
                "metadata": {"family": "instantsfm", "module": "instantsfm.scripts.sfm"},
            }
        ]

    def runtime_versions(self) -> dict[str, Any]:
        root = self._find_root()
        versions: dict[str, Any] = {
            "backend": self.version,
            "instantsfm_root": str(root) if root else "missing",
            "instantsfm_python": str(self._python_executable),
        }
        if root is not None:
            commit = self._git_revision(root)
            if commit:
                versions["instantsfm_commit"] = commit
        versions.update(self._torch_runtime_versions())
        return versions

    def _torch_runtime_versions(self) -> dict[str, Any]:
        try:
            import torch
        except Exception as exc:
            return {
                "torch_status": "unavailable",
                "torch_error": f"{type(exc).__name__}: {exc}",
            }

        out: dict[str, Any] = {
            "torch_status": "available",
            "torch_version": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda or "cpu"),
        }
        try:
            cuda_available = bool(torch.cuda.is_available())
            out["torch_cuda_available"] = cuda_available
            out["torch_cuda_device_count"] = int(torch.cuda.device_count()) if cuda_available else 0
            if cuda_available and out["torch_cuda_device_count"]:
                out["torch_cuda_device_name"] = str(torch.cuda.get_device_name(0))
        except Exception as exc:
            out["torch_cuda_available"] = False
            out["torch_cuda_device_count"] = 0
            out["torch_cuda_error"] = f"{type(exc).__name__}: {exc}"
        return out

    def list_backend_actions(self, *, include_schemas: bool = False) -> list[dict[str, Any]]:
        actions = [self._pipeline_action(include_schemas=include_schemas)]
        actions.extend(
            self._command_action(command, include_schemas=include_schemas)
            for command in INSTANTSFM_COMMANDS
        )
        actions.append(self._module_action(include_schemas=include_schemas))
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
        if action_id == "instantsfm.runPipeline":
            return self._run_pipeline(normalized, workspace=workspace, progress=progress)
        if action_id == "instantsfm.runModule":
            return self._run_module_action(normalized)
        command = _COMMAND_BY_ACTION.get(action_id)
        if command is None:
            raise NotFoundError(f"Backend action {action_id!r} not found")
        return self._run_command(command, normalized, progress=progress)

    # ------------------------------------------------------------------
    # Portable sfmapi mapping stage (map.global).
    #
    # InstantSfM's ``scripts.sfm`` hard-codes a ``<data_path>/{images,
    # database.db,sparse}`` project layout (see controllers.data_reader.
    # ReadData). sfmapi instead supplies independent db / image / sparse
    # paths, so this wrapper stages a temp directory whose ``database.db``
    # and ``images`` entries link to the caller's paths, runs InstantSfM
    # against that staged root, then reads the COLMAP sparse model back
    # out of ``<tmp>/sparse/0``.
    # ------------------------------------------------------------------

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
        """Run InstantSfM global mapping via a path-staging adapter.

        Returns ``(summaries, reconstructions)`` to satisfy the portable
        :class:`MappingBackend` protocol. InstantSfM emits a COLMAP
        sparse model directory; reading it back into a portable
        reconstruction object would need a reconstruction reader this
        wrapper does not implement, so the reconstruction list is left
        empty and the summary carries the on-disk ``model_path`` -- the
        same shape the SphereSfM backend's ``run_mapping`` returns.
        """
        normalized = str(kind).replace("-", "_").lower()
        if normalized != "global":
            raise CapabilityUnavailableError(
                capability=f"map.{kind}",
                reason="InstantSfM only implements portable global mapping (map.global).",
            )
        self._require_root()
        db_path = Path(db_path)
        image_root = Path(image_root)
        sparse_root = Path(sparse_root)
        job_dir = Path(job_dir)
        if not db_path.exists():
            raise ValidationError(f"InstantSfM mapping input database not found: {db_path}")
        if not image_root.exists():
            raise ValidationError(f"InstantSfM mapping image root not found: {image_root}")
        sparse_root.mkdir(parents=True, exist_ok=True)
        job_dir.mkdir(parents=True, exist_ok=True)

        # Stage <job_dir>/instantsfm_stage/{database.db,images} linking to
        # the caller's paths so scripts.sfm sees its expected layout.
        stage_root = job_dir / "instantsfm_stage"
        if stage_root.exists():
            shutil.rmtree(stage_root, ignore_errors=True)
        stage_root.mkdir(parents=True, exist_ok=True)
        staged_db = stage_root / "database.db"
        staged_images = stage_root / "images"
        db_link_mode = self._stage_link(db_path, staged_db)
        images_link_mode = self._stage_link(image_root, staged_images)

        module_args = ["--data_path", str(stage_root)]
        spec = dict(spec or {})
        if spec.get("export_txt") or spec.get("export_text"):
            module_args.append("--export_txt")
        if spec.get("disable_depths"):
            module_args.append("--disable_depths")
        if spec.get("disable_semantics"):
            module_args.append("--disable_semantics")
        manual_config = spec.get("manual_config_name")
        if manual_config:
            module_args.extend(["--manual_config_name", str(manual_config)])

        self._progress(progress, "global_mapping", 0, 1)
        try:
            completed = self._run_python_module(
                "instantsfm.scripts.sfm",
                module_args,
                timeout_seconds=spec.get("timeout_seconds"),
            )
            self._progress(progress, "global_mapping", 1, 1)

            # InstantSfM's reconstruction_writer.ExportReconstruction writes
            # the model into <data_path>/sparse/0.
            staged_sparse = stage_root / "sparse"
            produced = (
                sorted(
                    (path for path in staged_sparse.iterdir() if path.is_dir()),
                    key=lambda path: path.name,
                )
                if staged_sparse.is_dir()
                else []
            )
            summaries: list[dict[str, Any]] = []
            for model_dir in produced:
                # Move each staged sub-model under the caller's sparse_root so
                # it outlives the staging directory.
                dest = sparse_root / model_dir.name
                if dest.exists():
                    shutil.rmtree(dest, ignore_errors=True)
                shutil.move(str(model_dir), str(dest))
                model_name = model_dir.name
                summaries.append(
                    {
                        "idx": int(model_name) if model_name.isdigit() else model_name,
                        "model_path": str(dest),
                        "engine": "instantsfm scripts.sfm",
                    }
                )
        finally:
            # Always remove the staging tree — symlinks/junctions, or a full
            # image copy on the Windows fallback — even when the module run or
            # harvest raises. Previously this only ran on success and leaked
            # the staged image set on failure.
            shutil.rmtree(stage_root, ignore_errors=True)

        if not summaries:
            raise ValidationError(
                "InstantSfM global mapping produced no sparse model under "
                f"{staged_sparse} (stdout tail: {completed.stdout[-500:]!r})"
            )
        summaries[0].setdefault(
            "command",
            {
                "module": "instantsfm.scripts.sfm",
                "args": [
                    str(self._python_executable),
                    "-m",
                    "instantsfm.scripts.sfm",
                    *module_args,
                ],
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "staging": {
                    "stage_root": str(stage_root),
                    "database_link_mode": db_link_mode,
                    "images_link_mode": images_link_mode,
                },
            },
        )
        return summaries, []

    def _stage_link(self, source: Path, link: Path) -> str:
        """Link ``link`` -> ``source``, falling back to a copy.

        Prefers a symlink (works for both files and directories on Linux
        and on Windows when the process is allowed to create symlinks).
        On Windows without that privilege, a directory junction is used
        for directories; if every link strategy fails, the source is
        copied. Returns the strategy used: ``symlink`` | ``junction`` |
        ``copy``.
        """
        source = Path(source)
        # 1. Symlink -- cross-platform, cheapest, works for files + dirs.
        try:
            link.symlink_to(source, target_is_directory=source.is_dir())
            return "symlink"
        except (OSError, NotImplementedError):
            pass
        # 2. Windows directory junction -- no special privilege required.
        if os.name == "nt" and source.is_dir():
            try:
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(link), str(source)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return "junction"
            except (OSError, subprocess.CalledProcessError):
                pass
        # 3. Copy fallback -- always works, just slower / uses disk.
        if source.is_dir():
            shutil.copytree(source, link)
        else:
            link.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, link)
        return "copy"

    def _find_root(self) -> Path | None:
        if self._root_override is not None:
            if (self._root_override / "pyproject.toml").exists():
                return self._root_override
            return None
        return resolve_instantsfm_root(None)

    def _upstream_root_kind(self) -> str:
        """Portable identifier for where the InstantSfM source is rooted.

        Returned in action `metadata` instead of the absolute path the
        backend previously emitted, so framework snapshots (and the
        sfmapi-cpp port's generated backend_actions.inc) do not vary
        across machines. The absolute path is still available at
        runtime via ``/v1/admin/plugins/instantsfm:doctor``.
        """
        if self._root_override is not None:
            return "user-configured"
        root = self._find_root()
        if root is None:
            return "missing"
        if root == DEFAULT_INSTANTSFM_ROOT:
            return "bundled"
        return "discovered"

    def _require_root(self) -> Path:
        root = self._find_root()
        if root is None:
            raise CapabilityUnavailableError(
                capability="backend.actions",
                reason=(
                    "InstantSfM checkout not found. Run `git submodule update --init "
                    "--recursive` and set SFMAPI_INSTANTSFM_ROOT if needed."
                ),
            )
        return root

    def _git_revision(self, root: Path) -> str | None:
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

    def _run_pipeline(
        self,
        inputs: dict[str, Any],
        *,
        workspace: Path | None,
        progress: Any | None,
    ) -> dict[str, Any]:
        if workspace is not None:
            workspace.mkdir(parents=True, exist_ok=True)
        steps: list[tuple[InstantSfMCommand, dict[str, Any]]] = [
            (_COMMAND_BY_ACTION["instantsfm.extractFeatures"], inputs),
            (_COMMAND_BY_ACTION["instantsfm.runGlobalSfm"], inputs),
        ]
        if inputs.get("run_gaussian_splatting"):
            steps.append((_COMMAND_BY_ACTION["instantsfm.trainGaussianSplatting"], inputs))

        results: list[dict[str, Any]] = []
        total = len(steps)
        for index, (command, command_inputs) in enumerate(steps, start=1):
            self._progress(progress, command.category, index - 1, total)
            results.append(self._run_command(command, command_inputs, progress=None))
            self._progress(progress, command.category, index, total)
        return {
            "steps": results,
            "data_path": str(inputs["data_path"]),
            "sparse_path": str(Path(str(inputs["data_path"])) / "sparse"),
        }

    def _run_command(
        self,
        command: InstantSfMCommand,
        inputs: dict[str, Any],
        *,
        progress: Any | None,
    ) -> dict[str, Any]:
        module_args = self._module_args(command.action_id, inputs)
        self._progress(progress, command.category, 0, 1)
        completed = self._run_python_module(
            command.module,
            module_args,
            timeout_seconds=inputs.get("timeout_seconds"),
        )
        self._progress(progress, command.category, 1, 1)
        return {
            "action_id": command.action_id,
            "module": command.module,
            "args": [str(self._python_executable), "-m", command.module, *module_args],
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    def _run_module_action(self, inputs: dict[str, Any]) -> dict[str, Any]:
        module = str(inputs["module"])
        args = [str(arg) for arg in inputs.get("args", [])]
        completed = self._run_python_module(
            module,
            args,
            cwd=Path(str(inputs["cwd"])) if inputs.get("cwd") else None,
            extra_env={str(k): str(v) for k, v in dict(inputs.get("env") or {}).items()},
            timeout_seconds=inputs.get("timeout_seconds"),
        )
        return {
            "module": module,
            "args": [str(self._python_executable), "-m", module, *args],
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    def _run_python_module(
        self,
        module: str,
        args: list[str],
        *,
        cwd: Path | None = None,
        extra_env: dict[str, str] | None = None,
        timeout_seconds: int | float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        root = self._require_root()
        env = os.environ.copy()
        env["PYTHONPATH"] = instantsfm_pythonpath(root, env.get("PYTHONPATH"))
        if extra_env:
            env.update(extra_env)
        try:
            return subprocess.run(
                [str(self._python_executable), "-m", module, *args],
                check=True,
                capture_output=True,
                text=True,
                cwd=str(cwd or root),
                env=env,
                timeout=timeout_seconds,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            raise ValidationError(f"InstantSfM command failed: {detail}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ValidationError(f"InstantSfM command timed out after {timeout_seconds}s") from exc

    def _module_args(self, action_id: str, inputs: dict[str, Any]) -> list[str]:
        if action_id == "instantsfm.extractFeatures":
            args = ["--data_path", str(inputs["data_path"])]
            self._add_optional(args, inputs, "manual_config_name")
            self._add_optional(args, inputs, "feature_handler")
            self._add_flag(args, inputs, "single_camera")
            self._add_flag(args, inputs, "camera_per_folder")
            return args
        if action_id == "instantsfm.runGlobalSfm":
            args = ["--data_path", str(inputs["data_path"])]
            self._add_optional(args, inputs, "manual_config_name")
            self._add_optional(args, inputs, "record_path")
            for flag in (
                "enable_gui",
                "record_recon",
                "disable_depths",
                "disable_semantics",
                "export_txt",
            ):
                self._add_flag(args, inputs, flag)
            return args
        if action_id == "instantsfm.trainGaussianSplatting":
            return ["--data_path", str(inputs["data_path"])]
        if action_id == "instantsfm.visualizeReconstruction":
            args = ["--data_path", str(inputs["data_path"])]
            self._add_optional(args, inputs, "record")
            return args
        raise NotFoundError(f"Backend action {action_id!r} not found")

    def _pipeline_action(self, *, include_schemas: bool) -> dict[str, Any]:
        descriptor = {
            "action_id": "instantsfm.runPipeline",
            "backend": self.name,
            "display_name": "InstantSfM feature + global SfM pipeline",
            "description": "Run feature extraction, global SfM, and optional 3DGS training.",
            "category": "pipeline",
            "stability": "backend_extension",
            "side_effects": "write",
            "long_running": True,
            "supports_progress": True,
            "idempotent": False,
            "gpu_required": True,
            "required_capabilities": [],
            "metadata": {
                "family": "instantsfm",
                "upstream_root_kind": self._upstream_root_kind(),
            },
        }
        if include_schemas:
            descriptor["input_schema"] = self._pipeline_input_schema()
            descriptor["output_schema"] = self._run_output_schema()
        return descriptor

    def _command_action(
        self,
        command: InstantSfMCommand,
        *,
        include_schemas: bool,
    ) -> dict[str, Any]:
        descriptor = {
            "action_id": command.action_id,
            "backend": self.name,
            "display_name": command.display_name,
            "description": command.description,
            "category": command.category,
            "stability": "backend_extension",
            "side_effects": "write",
            "long_running": True,
            "supports_progress": False,
            "idempotent": False,
            "gpu_required": command.gpu_required,
            "required_capabilities": [],
            "metadata": {
                "family": "instantsfm",
                "module": command.module,
                "upstream_root_kind": self._upstream_root_kind(),
            },
        }
        if command.action_id == "instantsfm.trainGaussianSplatting":
            descriptor["metadata"]["optional_dependency_group"] = "gaussian_splatting"
            descriptor["metadata"]["provision_env"] = "SFMAPI_INSTANTSFM_INSTALL_GS=1"
        if include_schemas:
            descriptor["input_schema"] = self._input_schema_for_action(command.action_id)
            descriptor["output_schema"] = self._run_output_schema()
        return descriptor

    def _module_action(self, *, include_schemas: bool) -> dict[str, Any]:
        descriptor = {
            "action_id": "instantsfm.runModule",
            "backend": self.name,
            "display_name": "InstantSfM Python module",
            "description": "Run an allow-listed InstantSfM Python module with explicit args.",
            "category": "utility",
            "stability": "backend_extension",
            "side_effects": "write",
            "long_running": True,
            "supports_progress": False,
            "idempotent": False,
            "gpu_required": True,
            "required_capabilities": [],
            "metadata": {"family": "instantsfm", "allowlist": sorted(_SCRIPT_MODULES)},
        }
        if include_schemas:
            descriptor["input_schema"] = self._module_input_schema()
            descriptor["output_schema"] = self._run_output_schema()
        return descriptor

    def _input_schema_for_action(self, action_id: str) -> dict[str, Any]:
        base = self._common_input_schema()
        if action_id == "instantsfm.extractFeatures":
            base["properties"].update(
                {
                    "manual_config_name": {"type": "string"},
                    "feature_handler": {"type": "string", "enum": list(FEATURE_HANDLERS)},
                    "single_camera": {"type": "boolean", "default": False},
                    "camera_per_folder": {"type": "boolean", "default": False},
                }
            )
        elif action_id == "instantsfm.runGlobalSfm":
            base["properties"].update(
                {
                    "manual_config_name": {"type": "string"},
                    "enable_gui": {"type": "boolean", "default": False},
                    "record_recon": {"type": "boolean", "default": False},
                    "record_path": {"type": "string"},
                    "disable_depths": {"type": "boolean", "default": False},
                    "disable_semantics": {"type": "boolean", "default": False},
                    "export_txt": {"type": "boolean", "default": False},
                }
            )
        elif action_id == "instantsfm.visualizeReconstruction":
            base["properties"]["record"] = {"type": "string"}
        return base

    def _pipeline_input_schema(self) -> dict[str, Any]:
        schema = self._input_schema_for_action("instantsfm.extractFeatures")
        schema["properties"].update(
            self._input_schema_for_action("instantsfm.runGlobalSfm")["properties"]
        )
        schema["properties"]["run_gaussian_splatting"] = {"type": "boolean", "default": False}
        return schema

    def _common_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["data_path"],
            "properties": {
                "data_path": {"type": "string"},
                "timeout_seconds": {"type": "number"},
            },
        }

    def _module_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["module"],
            "properties": {
                "module": {"type": "string", "enum": sorted(_SCRIPT_MODULES)},
                "args": {"type": "array", "items": {"type": "string"}},
                "cwd": {"type": "string"},
                "env": {
                    "type": "object",
                    "additionalProperties": {"type": ["string", "number", "boolean"]},
                },
                "timeout_seconds": {"type": "number"},
            },
        }

    def _run_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "returncode": {"type": "integer"},
                "args": {"type": "array", "items": {"type": "string"}},
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
            },
        }

    def _normalize_action_inputs(self, action_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
        self.get_backend_action(action_id)
        if action_id == "instantsfm.runModule":
            module = str(inputs.get("module") or "")
            if module not in _SCRIPT_MODULES:
                raise ValidationError(
                    f"module must be one of: {', '.join(sorted(_SCRIPT_MODULES))}"
                )
            args = inputs.get("args", [])
            if args is None:
                args = []
            if not isinstance(args, list):
                raise ValidationError("args must be an array of strings")
            inputs["args"] = [str(arg) for arg in args]
            return inputs

        if not inputs.get("data_path"):
            raise ValidationError("data_path is required")
        if "feature_handler" in inputs:
            feature_handler = str(inputs["feature_handler"])
            if feature_handler not in FEATURE_HANDLERS:
                raise ValidationError(
                    f"feature_handler must be one of: {', '.join(FEATURE_HANDLERS)}"
                )
            inputs["feature_handler"] = feature_handler
        return inputs

    def _add_optional(self, args: list[str], inputs: dict[str, Any], name: str) -> None:
        value = inputs.get(name)
        if value is not None and str(value) != "":
            args.extend([f"--{name}", str(value)])

    def _add_flag(self, args: list[str], inputs: dict[str, Any], name: str) -> None:
        if bool(inputs.get(name, False)):
            args.append(f"--{name}")

    def _progress(self, progress: Any | None, phase: str, current: int, total: int) -> None:
        if progress is None:
            return
        try:
            progress.phase_progress(f"instantsfm.{phase}", current=current, total=total)
        except Exception:
            return


__all__ = [
    "DEFAULT_INSTANTSFM_ROOT",
    "FEATURE_HANDLERS",
    "INSTANTSFM_COMMANDS",
    "SKSPARSE_SHIM_DIR",
    "InstantSfMBackend",
    "configure_instantsfm_environment",
    "instantsfm_pythonpath",
    "resolve_instantsfm_root",
]
