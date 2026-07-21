from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from sceneapi.errors import CapabilityUnavailableError, NotFoundError, ValidationError
except ModuleNotFoundError:  # pragma: no cover - allows package tests without sfmapi installed

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

# Curated Reality CLI surface. This catalog covers the full photogrammetry
# workflow RealityScan / RealityCapture exposes through the CLI: alignment,
# reconstruction, texturing, meshing/model ops, ortho/DEM, and point-cloud
# export. Genuine UI-only, delegation, scene-view, upload, and paid add-on
# verbs (``uploadToSketchfab``, ``delegateTo``, scene-view navigation,
# ``dtmClassify``) stay out of the public action catalog because they are not
# headless photogrammetry workflow steps. ``execRSCMD`` is likewise excluded:
# it runs an arbitrary external command file, which would re-open the entire
# command surface this allow-list deliberately constrains.
REALITYSCAN_COMMAND_CATEGORIES: dict[str, tuple[str, ...]] = {
    "project_io": (
        "newScene",
        "load",
        "save",
        "start",
        "unlockPPIProject",
        "add",
        "addFolder",
        "importVideo",
        "importHDRimages",
        "addImageWithCalibration",
        "importImageSelection",
        "selectImage",
        "selectAllImages",
        "deselectAllImages",
        "invertImageSelection",
        "quit",
    ),
    "input_control": (
        "setFeatureSource",
        "enableAlignment",
        "enableMeshing",
        "setDownscaleForDepthMaps",
        "enableInComponent",
        "setCalibrationGroupByExif",
        "setConstantCalibrationGroups",
        "lockPoseForContinue",
        "setPriorCalibrationGroup",
        "setPriorLensGroup",
        "editInputSelection",
    ),
    "alignment": (
        "align",
        "draft",
        "update",
        "detectFeatures",
        "mergeComponents",
        "exportXMP",
        "exportXMPForSelectedComponent",
        "importComponent",
        "importBundler",
        "importColmap",
        "exportLatestComponents",
        "setMinComponentSize",
        "exportSelectedComponentDir",
        "exportSelectedComponentFile",
        "exportRegistration",
        "exportUndistortedImages",
        "exportSTMap",
        "exportSparsePointCloud",
        "selectComponent",
        "selectMaximalComponent",
        "selectComponentWithLeastReprojectionError",
    ),
    "control_points": (
        "importGroundControlPoints",
        "importControlPointsMeasurements",
        "listControlPoints",
        "selectControlPoint",
        "invertControlPointSelection",
        "selectMeasurementByError",
        "selectMeasurementByIndex",
        "exportGroundControlPoints",
        "exportControlPointsMeasurements",
        "defineDistance",
        "detectMarkers",
        "setCamerasGravityDirection",
    ),
    "reconstruction": (
        "resetGround",
        "setGroundPlaneFromReconstructionRegion",
        "setReconstructionRegionAuto",
        "setReconstructionRegion",
        "setReconstructionRegionOnCPs",
        "setReconstructionRegionByDensity",
        "scaleReconstructionRegion",
        "moveReconstructionRegion",
        "rotateReconstructionRegion",
        "offsetReconstructionRegion",
        "exportReconstructionRegion",
        "calculatePreviewModel",
        "calculateNormalModel",
        "calculateHighModel",
        "continueModelCalculation",
    ),
    "meshing": (
        "simplify",
        "simplifyOnReconstructionRegion",
        "smooth",
        "cleanModel",
        "closeHoles",
        "renameSelectedModel",
        "calculateTexturedModel",
    ),
    "texturing": (
        "unwrap",
        "calculateTexture",
        "calculateVertexColors",
        "reprojectTexture",
        "removeTexture",
        "colorizeModel",
        "addTextureLayer",
        "selectTextureLayer",
        "removeTextureLayer",
    ),
    "ortho_dem": (
        "defineOrthoProjection",
        "calculateOrthoProjection",
        "exportOrthoProjection",
        "exportDEM",
    ),
    "point_cloud_export": (
        "exportPointCloud",
        "exportPointCloudInBlocks",
    ),
    "model_export": (
        "selectModel",
        "selectLargestModelComponent",
        "exportModel",
        "exportSelectedModel",
        "exportModelToZip",
        "importModel",
    ),
    "settings": (
        "set",
        "preset",
        "reset",
        "writeProgress",
        "printProgress",
        "stdConsole",
        "disableOnlineCommunication",
        "importGlobalSettings",
        "exportGlobalSettings",
        "setProjectCoordinateSystem",
        "setOutputCoordinateSystem",
    ),
}

REALITYSCAN_COMMANDS: tuple[str, ...] = tuple(
    command for commands in REALITYSCAN_COMMAND_CATEGORIES.values() for command in commands
)
REALITYSCAN_COMMAND_SET = frozenset(REALITYSCAN_COMMANDS)
_COMMAND_TO_CATEGORY = {
    command: category
    for category, commands in REALITYSCAN_COMMAND_CATEGORIES.items()
    for command in commands
}

# GPU-bound CLI verbs: alignment, depth/model calculation, texturing, and
# ortho/DEM rasterization all run on the GPU. Every entry must also appear in
# ``REALITYSCAN_COMMANDS`` or it is unreachable through the action catalog.
_GPU_REQUIRED_COMMANDS: frozenset[str] = frozenset(
    {
        "align",
        "draft",
        "detectFeatures",
        "calculatePreviewModel",
        "calculateNormalModel",
        "calculateHighModel",
        "continueModelCalculation",
        "calculateTexturedModel",
        "unwrap",
        "calculateTexture",
        "calculateVertexColors",
        "reprojectTexture",
        "colorizeModel",
        "calculateOrthoProjection",
        "exportDEM",
    }
)
assert _GPU_REQUIRED_COMMANDS <= REALITYSCAN_COMMAND_SET, (
    "every GPU-required RealityScan verb must be present in REALITYSCAN_COMMANDS"
)

# Canned multi-step photogrammetry workflows exposed alongside the per-verb
# actions and ``realityscan.reconstructImageFolder``. Each maps to a builder
# method on ``RealityScanCliBackend`` that emits a fixed CLI command sequence.
_WORKFLOW_ACTIONS: tuple[dict[str, Any], ...] = (
    {
        "action_id": "realityscan.reconstructToTexturedMesh",
        "method": "reconstruct_to_textured_mesh",
        "display_suffix": "reconstruct to textured mesh",
        "description": (
            "Create a project from an image folder, then align, set the "
            "reconstruction region, calculate the high-detail model, unwrap, "
            "calculate texture, and export the textured model."
        ),
        "required": ("image_folder", "project_path", "export_model_path"),
        "extra_properties": {
            "image_folder": {"type": "string"},
            "project_path": {"type": "string"},
            "export_model_path": {"type": "string"},
            "export_params_path": {"type": "string"},
        },
    },
    {
        "action_id": "realityscan.reconstructToOrthophoto",
        "method": "reconstruct_to_orthophoto",
        "display_suffix": "reconstruct to orthophoto",
        "description": (
            "Create a project from an image folder, then align, calculate the "
            "model, calculate the ortho projection, and export the orthophoto."
        ),
        "required": ("image_folder", "project_path", "export_ortho_path"),
        "extra_properties": {
            "image_folder": {"type": "string"},
            "project_path": {"type": "string"},
            "export_ortho_path": {"type": "string"},
            "quality": {"type": "string", "enum": ["preview", "normal", "high"]},
        },
    },
    {
        "action_id": "realityscan.alignOnly",
        "method": "align_only",
        "display_suffix": "align only",
        "description": (
            "Create a project from an image folder, run alignment, and export "
            "the registration and/or sparse point cloud without meshing."
        ),
        "required": ("image_folder", "project_path"),
        "extra_properties": {
            "image_folder": {"type": "string"},
            "project_path": {"type": "string"},
            "export_registration_path": {"type": "string"},
            "export_sparse_point_cloud_path": {"type": "string"},
        },
    },
)
_WORKFLOW_ACTIONS_BY_ID = {workflow["action_id"]: workflow for workflow in _WORKFLOW_ACTIONS}


@dataclass(frozen=True)
class RealityCliInterface:
    interface_id: str
    product: str
    display_name: str
    executable_names: tuple[str, ...]
    install_dir_prefixes: tuple[str, ...]
    project_extensions: tuple[str, ...]
    command_file_extensions: tuple[str, ...]
    docs_url: str
    commands: tuple[str, ...] = REALITYSCAN_COMMANDS
    command_aliases: dict[str, str] | None = None

    @property
    def default_project_extension(self) -> str:
        return self.project_extensions[0]


@dataclass(frozen=True)
class RealityCliInstallation:
    executable: Path
    interface: RealityCliInterface
    version_hint: str


INTERFACE_LUT: dict[str, RealityCliInterface] = {
    "realitycapture.current": RealityCliInterface(
        interface_id="realitycapture.current",
        product="realitycapture",
        display_name="RealityCapture",
        executable_names=("RealityCapture.exe", "RealityCapture"),
        install_dir_prefixes=("RealityCapture",),
        project_extensions=(".rcproj", ".rsproj"),
        command_file_extensions=(".rccmd", ".rscmd"),
        docs_url="https://rchelp.capturingreality.com/en-US/appbasics/allcommands.htm",
    ),
    "realityscan.2.1": RealityCliInterface(
        interface_id="realityscan.2.1",
        product="realityscan",
        display_name="RealityScan 2.1+",
        executable_names=("RealityScan.exe", "RealityScan"),
        install_dir_prefixes=("RealityScan",),
        project_extensions=(".rsproj", ".rcproj"),
        command_file_extensions=(".rscmd", ".rccmd"),
        docs_url="https://rshelp.capturingreality.com/en-US/appbasics/allcommands.htm",
    ),
    "realityscan.current": RealityCliInterface(
        interface_id="realityscan.current",
        product="realityscan",
        display_name="RealityScan",
        executable_names=("RealityScan.exe", "RealityScan"),
        install_dir_prefixes=("RealityScan",),
        project_extensions=(".rsproj", ".rcproj"),
        command_file_extensions=(".rscmd", ".rccmd"),
        docs_url="https://rshelp.capturingreality.com/en-US/appbasics/allcommands.htm",
    ),
}

REALITY_CLI_ENV_VARS: tuple[str, ...] = (
    "SFMAPI_RC_EXECUTABLE",
    "SFMAPI_REALITYCAPTURE_EXECUTABLE",
    "SFMAPI_REALITYSCAN_EXECUTABLE",
)


def _realityscan_executable_names() -> list[str]:
    names: list[str] = []
    # Prefer the system RealityCapture CLI when present. RealityScan is the fallback.
    for interface_id in ("realitycapture.current", "realityscan.2.1"):
        names.extend(INTERFACE_LUT[interface_id].executable_names)
    return names


def _configured_realityscan_candidates(value: str | Path) -> list[Path]:
    raw = os.path.expandvars(str(value)).strip().strip('"')
    path = Path(raw).expanduser()
    if path.is_dir():
        candidates: list[Path] = []
        for name in _realityscan_executable_names():
            candidates.extend([path / name, path / "bin" / name])
        return candidates
    return [path]


def resolve_realityscan_executable(value: str | Path | None) -> Path | None:
    if not value:
        return None
    for candidate in _configured_realityscan_candidates(value):
        if candidate.exists():
            return candidate.resolve()
    return None


def _infer_cli_interface(executable: Path) -> RealityCliInterface:
    lower = " ".join(part.lower() for part in [executable.name, executable.parent.name])
    if "realitycapture" in lower:
        return INTERFACE_LUT["realitycapture.current"]
    if "realityscan" in lower:
        version = _version_tuple_from_name(executable.parent.name)
        if version >= (2, 1):
            return INTERFACE_LUT["realityscan.2.1"]
        return INTERFACE_LUT["realityscan.current"]
    return (
        INTERFACE_LUT["realitycapture.current"]
        if os.name == "nt"
        else INTERFACE_LUT["realityscan.current"]
    )


def _version_tuple_from_name(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", value))


def resolve_reality_cli_installation(value: str | Path | None) -> RealityCliInstallation | None:
    executable = resolve_realityscan_executable(value)
    if executable is None:
        return None
    return RealityCliInstallation(
        executable=executable,
        interface=_infer_cli_interface(executable),
        version_hint=".".join(
            str(part) for part in _version_tuple_from_name(str(executable.parent))
        )
        or "unknown",
    )


def realityscan_runtime_dirs(executable: str | Path) -> list[Path]:
    exe = Path(executable).resolve()
    candidates = [exe.parent, exe.parent / "bin", exe.parent.parent / "bin"]
    seen: set[Path] = set()
    dirs: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.exists() or not resolved.is_dir():
            continue
        seen.add(resolved)
        dirs.append(resolved)
    return dirs


def _versioned_install_dirs(root: Path, prefix: str) -> list[Path]:
    def sort_key(path: Path) -> tuple[tuple[int, ...], str]:
        version = _version_tuple_from_name(path.name)
        return version, path.name.lower()

    return sorted(root.glob(f"{prefix}*"), key=sort_key, reverse=True)


def _windows_app_path_candidates() -> list[Path]:
    if os.name != "nt":
        return []
    try:
        import winreg
    except ImportError:
        return []

    candidates: list[Path] = []
    subkeys = [
        rf"Software\Microsoft\Windows\CurrentVersion\App Paths\{name}"
        for name in _realityscan_executable_names()
        if name.endswith(".exe")
    ]
    for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        for subkey in subkeys:
            try:
                with winreg.OpenKey(root, subkey) as key:
                    value, _ = winreg.QueryValueEx(key, "")
            except OSError:
                continue
            if value:
                candidates.append(Path(str(value)))
    return candidates


def _default_install_candidates() -> list[Path]:
    candidates: list[Path] = []
    candidates.extend(_windows_app_path_candidates())
    if os.name == "nt":
        roots = [
            os.environ.get("PROGRAMFILES"),
            os.environ.get("PROGRAMFILES(X86)"),
            os.environ.get("LOCALAPPDATA"),
        ]
        for root in roots:
            if not root:
                continue
            epic_root = Path(root) / "Epic Games"
            capturing_reality_root = Path(root) / "Capturing Reality"
            # Prefer RealityCapture when both products are installed; it is the system RC CLI.
            candidates.extend(
                [
                    epic_root / "RealityCapture" / "RealityCapture.exe",
                    epic_root / "RealityScan" / "RealityScan.exe",
                    capturing_reality_root / "RealityCapture" / "RealityCapture.exe",
                    capturing_reality_root / "RealityScan" / "RealityScan.exe",
                ]
            )
            if epic_root.exists():
                for install_dir in _versioned_install_dirs(epic_root, "RealityCapture"):
                    candidates.append(install_dir / "RealityCapture.exe")
                for install_dir in _versioned_install_dirs(epic_root, "RealityScan"):
                    candidates.append(install_dir / "RealityScan.exe")
            if capturing_reality_root.exists():
                for install_dir in _versioned_install_dirs(
                    capturing_reality_root, "RealityCapture"
                ):
                    candidates.append(install_dir / "RealityCapture.exe")
                for install_dir in _versioned_install_dirs(capturing_reality_root, "RealityScan"):
                    candidates.append(install_dir / "RealityScan.exe")
    return candidates


def _configured_executable() -> str | None:
    for env_var in REALITY_CLI_ENV_VARS:
        value = os.environ.get(env_var)
        if value:
            return value
    return None


def configure_realityscan_environment(
    executable: str | Path | None = None,
    *,
    validate: bool = False,
) -> Path | None:
    installation = configure_reality_cli_environment(executable, validate=validate)
    return installation.executable if installation else None


def configure_reality_cli_environment(
    executable: str | Path | None = None,
    *,
    validate: bool = False,
) -> RealityCliInstallation | None:
    configured = executable or _configured_executable()
    installation = resolve_reality_cli_installation(configured)
    if configured and installation is None:
        if validate:
            raise ValueError(
                "RealityCapture/RealityScan executable not found at the configured path. "
                "Set SFMAPI_RC_EXECUTABLE, SFMAPI_REALITYCAPTURE_EXECUTABLE, "
                "or SFMAPI_REALITYSCAN_EXECUTABLE to the executable or install directory."
            )
        return None
    resolved = installation.executable if installation else None
    if resolved is None:
        for name in _realityscan_executable_names():
            found = shutil.which(name)
            if found:
                resolved = Path(found).resolve()
                installation = RealityCliInstallation(
                    executable=resolved,
                    interface=_infer_cli_interface(resolved),
                    version_hint="path",
                )
                break
    if resolved is None:
        for candidate in _default_install_candidates():
            if candidate.exists():
                resolved = candidate.resolve()
                installation = RealityCliInstallation(
                    executable=resolved,
                    interface=_infer_cli_interface(resolved),
                    version_hint=".".join(
                        str(part) for part in _version_tuple_from_name(str(resolved.parent))
                    )
                    or "system",
                )
                break
    if resolved is None:
        return None

    assert installation is not None
    os.environ["SFMAPI_REALITYSCAN_EXECUTABLE"] = str(resolved)
    os.environ["SFMAPI_RC_EXECUTABLE"] = str(resolved)
    if installation.interface.product == "realitycapture":
        os.environ["SFMAPI_REALITYCAPTURE_EXECUTABLE"] = str(resolved)
    else:
        os.environ["SFMAPI_REALITYSCAN_EXECUTABLE"] = str(resolved)
    existing = os.environ.get("PATH", "")
    existing_parts = [part for part in existing.split(os.pathsep) if part]
    existing_norm = {
        str(Path(part).resolve()).lower() for part in existing_parts if Path(part).exists()
    }
    prepended = [
        str(path)
        for path in realityscan_runtime_dirs(resolved)
        if str(path).lower() not in existing_norm
    ]
    if prepended:
        os.environ["PATH"] = os.pathsep.join([*prepended, existing])
    return installation


class RealityScanCliBackend:
    name = "realityscan_cli"
    version = "0.0.1"
    vendor = "Epic Games RealityScan"

    def __init__(self, executable: str | Path | None = None) -> None:
        self._installation_override = (
            resolve_reality_cli_installation(executable) if executable else None
        )
        self._cached_installation = self._installation_override
        self._executable_override = (
            self._installation_override.executable if self._installation_override else None
        )

    def capabilities(self) -> set[str]:
        return set()

    def runtime_versions(self) -> dict[str, str]:
        installation = self._find_installation()
        exe = installation.executable if installation else None
        interface = (
            installation.interface if installation else INTERFACE_LUT["realitycapture.current"]
        )
        versions = {
            "backend": self.version,
            "reality_cli_executable": str(exe) if exe else "missing",
            "reality_cli_interface": interface.interface_id,
            "reality_cli_product": interface.product,
            "reality_cli_project_extensions": ",".join(interface.project_extensions),
        }
        if exe is not None:
            try:
                stat = exe.stat()
            except OSError:
                pass
            else:
                versions["reality_cli_executable_size"] = str(stat.st_size)
                versions["reality_cli_executable_mtime_ns"] = str(stat.st_mtime_ns)
            # Backwards-compatible key for callers already using this demo.
            versions["realityscan_executable"] = str(exe)
        return versions

    def list_backend_artifact_contracts(self) -> list[dict[str, Any]]:
        """Action-only artifact contracts for RealityScan alignment exports.

        RealityScan exposes nothing through the portable capability
        vocabulary (``capabilities()`` is empty), so these contracts carry
        ``capability=None``. They describe the registration and sparse
        point-cloud artifacts produced by the ``exportRegistration`` and
        ``exportSparsePointCloud`` CLI verbs (and the ``alignOnly`` workflow).
        """
        interface = self._interface()
        return [
            {
                "contract_id": "realityscan.registration",
                "backend": self.name,
                "stage": "mapping",
                "capability": None,
                "provider": self.name,
                "display_name": f"{interface.display_name} registration export",
                "description": (
                    "Camera registration (poses, calibration, sparse tie points) "
                    "written by the `-exportRegistration` CLI verb."
                ),
                "accepts": [],
                "emits": ["reconstruction.sparse.v1"],
                "preferred": "reconstruction.sparse.v1",
                "metadata": {
                    "family": "reality_cli",
                    "product": interface.product,
                    "interface_id": interface.interface_id,
                    "command": "exportRegistration",
                    "docs_url": interface.docs_url,
                },
            },
            {
                "contract_id": "realityscan.sparse_point_cloud",
                "backend": self.name,
                "stage": "mapping",
                "capability": None,
                "provider": self.name,
                "display_name": f"{interface.display_name} sparse point cloud export",
                "description": (
                    "Sparse alignment point cloud written by the "
                    "`-exportSparsePointCloud` CLI verb."
                ),
                "accepts": [],
                "emits": ["reconstruction.sparse.v1"],
                "preferred": "reconstruction.sparse.v1",
                "metadata": {
                    "family": "reality_cli",
                    "product": interface.product,
                    "interface_id": interface.interface_id,
                    "command": "exportSparsePointCloud",
                    "docs_url": interface.docs_url,
                },
            },
        ]

    def list_backend_actions(self, *, include_schemas: bool = False) -> list[dict[str, Any]]:
        interface = self._interface()
        actions = [self._reconstruct_action(include_schemas=include_schemas)]
        actions.append(self._run_sequence_action(include_schemas=include_schemas))
        for workflow in _WORKFLOW_ACTIONS:
            actions.append(self._workflow_action(workflow, include_schemas=include_schemas))
        for command in interface.commands:
            actions.append(self._command_action(command, include_schemas=include_schemas))
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
        if action_id == "realityscan.runSequence":
            return self.run_realityscan_sequence(
                self._commands_from_inputs(normalized),
                headless=bool(normalized.get("headless", True)),
                std_console=bool(normalized.get("std_console", True)),
                fail_on_error=bool(normalized.get("fail_on_error", True)),
                silent_crash_report_path=normalized.get("silent_crash_report_path"),
                write_progress_path=normalized.get("write_progress_path"),
                progress_timeout_seconds=normalized.get("progress_timeout_seconds"),
                timeout_seconds=normalized.get("timeout_seconds"),
                progress=progress,
            )
        if action_id == "realityscan.reconstructImageFolder":
            return self.reconstruct_image_folder(
                image_folder=Path(str(normalized["image_folder"])),
                project_path=Path(str(normalized["project_path"])),
                quality=str(normalized.get("quality", "normal")),
                export_model_path=(
                    Path(str(normalized["export_model_path"]))
                    if normalized.get("export_model_path")
                    else None
                ),
                export_params_path=(
                    Path(str(normalized["export_params_path"]))
                    if normalized.get("export_params_path")
                    else None
                ),
                settings=dict(normalized.get("settings") or {}),
                headless=bool(normalized.get("headless", True)),
                std_console=bool(normalized.get("std_console", True)),
                timeout_seconds=normalized.get("timeout_seconds"),
                workspace=workspace,
                progress=progress,
            )
        if action_id in _WORKFLOW_ACTIONS_BY_ID:
            return self._run_workflow_action(
                action_id, normalized, workspace=workspace, progress=progress
            )
        command = self._command_from_action_id(action_id)
        return self.run_realityscan_command(
            command,
            args=[str(arg) for arg in normalized.get("args", [])],
            headless=bool(normalized.get("headless", True)),
            std_console=bool(normalized.get("std_console", True)),
            fail_on_error=bool(normalized.get("fail_on_error", True)),
            append_quit=bool(normalized.get("append_quit", True)),
            silent_crash_report_path=normalized.get("silent_crash_report_path"),
            write_progress_path=normalized.get("write_progress_path"),
            progress_timeout_seconds=normalized.get("progress_timeout_seconds"),
            timeout_seconds=normalized.get("timeout_seconds"),
            progress=progress,
        )

    def run_realityscan_command(
        self,
        command: str,
        *,
        args: Sequence[str | Path] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._validate_command(command)
        return self.run_realityscan_sequence(
            [(command, [str(arg) for arg in args or []])], **kwargs
        )

    def reconstruct_image_folder(
        self,
        *,
        image_folder: Path,
        project_path: Path,
        quality: str = "normal",
        export_model_path: Path | None = None,
        export_params_path: Path | None = None,
        settings: dict[str, Any] | None = None,
        headless: bool = True,
        std_console: bool = True,
        timeout_seconds: int | float | None = None,
        workspace: Path | None = None,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        if not image_folder.exists() or not image_folder.is_dir():
            raise ValidationError(
                f"image_folder does not exist or is not a directory: {image_folder}"
            )
        quality_command = {
            "preview": "calculatePreviewModel",
            "normal": "calculateNormalModel",
            "high": "calculateHighModel",
        }.get(quality)
        if quality_command is None:
            raise ValidationError("quality must be one of: preview, normal, high")

        if not project_path.suffix:
            project_path = project_path.with_suffix(self._interface().default_project_extension)
        project_path.parent.mkdir(parents=True, exist_ok=True)
        if export_model_path is not None:
            export_model_path.parent.mkdir(parents=True, exist_ok=True)
        scratch = Path(workspace) if workspace is not None else project_path.parent
        scratch.mkdir(parents=True, exist_ok=True)
        progress_file = scratch / f"{project_path.stem}.realityscan-progress.txt"

        commands: list[tuple[str, list[str]]] = [("newScene", [])]
        for key, value in sorted((settings or {}).items()):
            commands.append(("set", [f"{key}={self._stringify(value)}"]))
        commands.extend(
            [
                ("addFolder", [str(image_folder)]),
                ("align", []),
                ("selectMaximalComponent", []),
                ("setReconstructionRegionAuto", []),
                (quality_command, []),
            ]
        )
        if export_model_path is not None:
            export_args = [str(export_model_path)]
            if export_params_path is not None:
                export_args.append(str(export_params_path))
            commands.append(("exportSelectedModel", export_args))
        commands.append(("save", [str(project_path)]))

        result = self.run_realityscan_sequence(
            commands,
            headless=headless,
            std_console=std_console,
            fail_on_error=True,
            append_quit=True,
            write_progress_path=progress_file,
            timeout_seconds=timeout_seconds,
            progress=progress,
        )
        result.update(
            {
                "image_folder": str(image_folder),
                "project_path": str(project_path),
                "quality": quality,
                "export_model_path": str(export_model_path) if export_model_path else None,
            }
        )
        return result

    def reconstruct_to_textured_mesh(
        self,
        *,
        image_folder: Path,
        project_path: Path,
        export_model_path: Path,
        export_params_path: Path | None = None,
        settings: dict[str, Any] | None = None,
        headless: bool = True,
        std_console: bool = True,
        timeout_seconds: int | float | None = None,
        workspace: Path | None = None,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        """Folder -> align -> region -> high model -> unwrap -> texture -> export."""
        project_path, progress_file = self._prepare_workflow_paths(project_path, workspace)
        export_model_path.parent.mkdir(parents=True, exist_ok=True)
        commands = self._workflow_prologue(image_folder, settings)
        commands.extend(
            [
                ("align", []),
                ("selectMaximalComponent", []),
                ("setReconstructionRegionAuto", []),
                ("calculateHighModel", []),
                ("unwrap", []),
                ("calculateTexture", []),
            ]
        )
        export_args = [str(export_model_path)]
        if export_params_path is not None:
            export_args.append(str(export_params_path))
        commands.append(("exportSelectedModel", export_args))
        commands.append(("save", [str(project_path)]))

        result = self.run_realityscan_sequence(
            commands,
            headless=headless,
            std_console=std_console,
            fail_on_error=True,
            append_quit=True,
            write_progress_path=progress_file,
            timeout_seconds=timeout_seconds,
            progress=progress,
        )
        result.update(
            {
                "workflow": "reconstruct_to_textured_mesh",
                "image_folder": str(image_folder),
                "project_path": str(project_path),
                "export_model_path": str(export_model_path),
            }
        )
        return result

    def reconstruct_to_orthophoto(
        self,
        *,
        image_folder: Path,
        project_path: Path,
        export_ortho_path: Path,
        quality: str = "normal",
        settings: dict[str, Any] | None = None,
        headless: bool = True,
        std_console: bool = True,
        timeout_seconds: int | float | None = None,
        workspace: Path | None = None,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        """Folder -> align -> mesh -> calculateOrthoProjection -> exportOrthoProjection."""
        quality_command = self._quality_command(quality)
        project_path, progress_file = self._prepare_workflow_paths(project_path, workspace)
        export_ortho_path.parent.mkdir(parents=True, exist_ok=True)
        commands = self._workflow_prologue(image_folder, settings)
        commands.extend(
            [
                ("align", []),
                ("selectMaximalComponent", []),
                ("setReconstructionRegionAuto", []),
                (quality_command, []),
                ("calculateOrthoProjection", []),
                ("exportOrthoProjection", [str(export_ortho_path)]),
                ("save", [str(project_path)]),
            ]
        )

        result = self.run_realityscan_sequence(
            commands,
            headless=headless,
            std_console=std_console,
            fail_on_error=True,
            append_quit=True,
            write_progress_path=progress_file,
            timeout_seconds=timeout_seconds,
            progress=progress,
        )
        result.update(
            {
                "workflow": "reconstruct_to_orthophoto",
                "image_folder": str(image_folder),
                "project_path": str(project_path),
                "quality": quality,
                "export_ortho_path": str(export_ortho_path),
            }
        )
        return result

    def align_only(
        self,
        *,
        image_folder: Path,
        project_path: Path,
        export_registration_path: Path | None = None,
        export_sparse_point_cloud_path: Path | None = None,
        settings: dict[str, Any] | None = None,
        headless: bool = True,
        std_console: bool = True,
        timeout_seconds: int | float | None = None,
        workspace: Path | None = None,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        """Folder -> align -> exportRegistration / exportSparsePointCloud."""
        if export_registration_path is None and export_sparse_point_cloud_path is None:
            raise ValidationError(
                "align_only requires export_registration_path or export_sparse_point_cloud_path"
            )
        project_path, progress_file = self._prepare_workflow_paths(project_path, workspace)
        commands = self._workflow_prologue(image_folder, settings)
        commands.extend([("align", []), ("selectMaximalComponent", [])])
        if export_registration_path is not None:
            export_registration_path.parent.mkdir(parents=True, exist_ok=True)
            commands.append(("exportRegistration", [str(export_registration_path)]))
        if export_sparse_point_cloud_path is not None:
            export_sparse_point_cloud_path.parent.mkdir(parents=True, exist_ok=True)
            commands.append(("exportSparsePointCloud", [str(export_sparse_point_cloud_path)]))
        commands.append(("save", [str(project_path)]))

        result = self.run_realityscan_sequence(
            commands,
            headless=headless,
            std_console=std_console,
            fail_on_error=True,
            append_quit=True,
            write_progress_path=progress_file,
            timeout_seconds=timeout_seconds,
            progress=progress,
        )
        result.update(
            {
                "workflow": "align_only",
                "image_folder": str(image_folder),
                "project_path": str(project_path),
                "export_registration_path": (
                    str(export_registration_path) if export_registration_path else None
                ),
                "export_sparse_point_cloud_path": (
                    str(export_sparse_point_cloud_path) if export_sparse_point_cloud_path else None
                ),
            }
        )
        return result

    def _run_workflow_action(
        self,
        action_id: str,
        normalized: dict[str, Any],
        *,
        workspace: Path | None,
        progress: Any | None,
    ) -> dict[str, Any]:
        common = {
            "settings": dict(normalized.get("settings") or {}),
            "headless": bool(normalized.get("headless", True)),
            "std_console": bool(normalized.get("std_console", True)),
            "timeout_seconds": normalized.get("timeout_seconds"),
            "workspace": workspace,
            "progress": progress,
        }
        image_folder = Path(str(normalized["image_folder"]))
        project_path = Path(str(normalized["project_path"]))
        if action_id == "realityscan.reconstructToTexturedMesh":
            return self.reconstruct_to_textured_mesh(
                image_folder=image_folder,
                project_path=project_path,
                export_model_path=Path(str(normalized["export_model_path"])),
                export_params_path=(
                    Path(str(normalized["export_params_path"]))
                    if normalized.get("export_params_path")
                    else None
                ),
                **common,
            )
        if action_id == "realityscan.reconstructToOrthophoto":
            return self.reconstruct_to_orthophoto(
                image_folder=image_folder,
                project_path=project_path,
                export_ortho_path=Path(str(normalized["export_ortho_path"])),
                quality=str(normalized.get("quality", "normal")),
                **common,
            )
        if action_id == "realityscan.alignOnly":
            return self.align_only(
                image_folder=image_folder,
                project_path=project_path,
                export_registration_path=(
                    Path(str(normalized["export_registration_path"]))
                    if normalized.get("export_registration_path")
                    else None
                ),
                export_sparse_point_cloud_path=(
                    Path(str(normalized["export_sparse_point_cloud_path"]))
                    if normalized.get("export_sparse_point_cloud_path")
                    else None
                ),
                **common,
            )
        raise NotFoundError(f"Backend action {action_id!r} not found")

    def _quality_command(self, quality: str) -> str:
        quality_command = {
            "preview": "calculatePreviewModel",
            "normal": "calculateNormalModel",
            "high": "calculateHighModel",
        }.get(quality)
        if quality_command is None:
            raise ValidationError("quality must be one of: preview, normal, high")
        return quality_command

    def _prepare_workflow_paths(
        self, project_path: Path, workspace: Path | None
    ) -> tuple[Path, Path]:
        """Resolve the project extension and return ``(project_path, progress_file)``."""
        if not project_path.suffix:
            project_path = project_path.with_suffix(self._interface().default_project_extension)
        project_path.parent.mkdir(parents=True, exist_ok=True)
        scratch = Path(workspace) if workspace is not None else project_path.parent
        scratch.mkdir(parents=True, exist_ok=True)
        progress_file = scratch / f"{project_path.stem}.realityscan-progress.txt"
        return project_path, progress_file

    def _workflow_prologue(
        self, image_folder: Path, settings: dict[str, Any] | None
    ) -> list[tuple[str, list[str]]]:
        if not image_folder.exists() or not image_folder.is_dir():
            raise ValidationError(
                f"image_folder does not exist or is not a directory: {image_folder}"
            )
        commands: list[tuple[str, list[str]]] = [("newScene", [])]
        for key, value in sorted((settings or {}).items()):
            commands.append(("set", [f"{key}={self._stringify(value)}"]))
        commands.append(("addFolder", [str(image_folder)]))
        return commands

    def run_realityscan_sequence(
        self,
        commands: Sequence[tuple[str, Sequence[str | Path]]],
        *,
        headless: bool = True,
        std_console: bool = True,
        fail_on_error: bool = True,
        append_quit: bool = True,
        silent_crash_report_path: str | Path | None = None,
        write_progress_path: str | Path | None = None,
        progress_timeout_seconds: int | float | None = None,
        timeout_seconds: int | float | None = None,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        exe = self._require_realityscan()
        cli_args = [str(exe)]
        if headless:
            cli_args.append("-headless")
        if std_console:
            cli_args.append("-stdConsole")
        if silent_crash_report_path:
            Path(silent_crash_report_path).mkdir(parents=True, exist_ok=True)
            cli_args.extend(["-silent", str(silent_crash_report_path)])
        if fail_on_error:
            cli_args.extend(["-set", "appQuitOnError=true", "-set", "suppressErrors=true"])
        if write_progress_path:
            progress_path = Path(write_progress_path)
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            cli_args.extend(["-writeProgress", str(progress_path)])
            if progress_timeout_seconds is not None:
                cli_args.append(self._stringify(progress_timeout_seconds))

        for command, command_args in commands:
            self._validate_command(command)
            cli_args.append(f"-{command}")
            cli_args.extend(str(arg) for arg in command_args)
        if append_quit and "quit" not in [command for command, _ in commands]:
            cli_args.append("-quit")

        completed = self._run(cli_args, timeout_seconds=timeout_seconds)
        progress_records = (
            self._read_progress_records(Path(write_progress_path)) if write_progress_path else []
        )
        self._emit_progress(progress, progress_records)
        return {
            "returncode": completed.returncode,
            "args": cli_args,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "progress_records": progress_records,
        }

    def extract_features(self, **_: Any) -> dict:
        raise self._unsupported(
            "features.extract.sift", "RealityScan CLI does not expose sfmapi feature DBs"
        )

    def match(self, **_: Any) -> dict:
        raise self._unsupported("pairs.exhaustive", "Use native RealityScan alignment actions")

    def verify_matches(self, **_: Any) -> dict:
        raise self._unsupported("matches.verify", "Use native RealityScan alignment actions")

    def read_keypoints(self, **_: Any) -> tuple[list[list[float]], bytes, int]:
        raise self._unsupported(
            "observations.by_image", "RealityScan CLI does not expose keypoints"
        )

    def iter_two_view_geometries(self, **_: Any) -> Iterable[tuple[int, int, Any]]:
        raise self._unsupported("matches.verify", "RealityScan CLI does not expose pair geometry")

    def iter_correspondences(self, **_: Any) -> Iterable[tuple[int, int, Any]]:
        raise self._unsupported("pairs.exhaustive", "RealityScan CLI does not expose raw matches")

    def run_mapping(self, **_: Any) -> tuple[list[dict], list[Any]]:
        raise self._unsupported("map.incremental", "Use realityscan.reconstructImageFolder")

    def bundle_adjustment(self, **_: Any) -> dict:
        raise self._unsupported("ba.standard", "Use native RealityScan project/model actions")

    def triangulate(self, **_: Any) -> dict:
        raise self._unsupported("triangulate.retri", "Use native RealityScan project/model actions")

    def relocalize(self, **_: Any) -> dict:
        raise self._unsupported("relocalize.images", "Use native RealityScan project/model actions")

    def pose_graph_optimize(self, **_: Any) -> dict:
        raise self._unsupported("pgo.optimize", "Use native RealityScan project/model actions")

    def export(self, **_: Any) -> dict:
        raise self._unsupported("export.ply", "Use realityscan.exportSelectedModel")

    def convert_spherical_to_cubemap(self, **_: Any) -> dict:
        raise self._unsupported("projection.cubemap_rig")

    def render_spherical_cubemap_images(self, **_: Any) -> dict:
        raise self._unsupported("projection.equirectangular_to_cubemap")

    def build_vlad_index(self, **_: Any) -> tuple[list[str], Any]:
        raise self._unsupported("similarity.vlad")

    def localize_from_memory(self, **_: Any) -> dict:
        raise self._unsupported("localize.from_memory")

    def apply_sim3(self, **_: Any) -> dict:
        raise self._unsupported(
            "georegister.sim3", "Use native RealityScan transform/export actions"
        )

    def read_reconstruction(self, path: Path) -> Any:
        raise self._unsupported(
            "export.colmap_text", f"Cannot parse RealityScan project as sfmapi model: {path}"
        )

    def _find_installation(self) -> RealityCliInstallation | None:
        if self._installation_override and self._installation_override.executable.exists():
            return self._installation_override
        if self._cached_installation and self._cached_installation.executable.exists():
            return self._cached_installation
        self._cached_installation = configure_reality_cli_environment(validate=False)
        return self._cached_installation

    def _interface(self) -> RealityCliInterface:
        installation = self._find_installation()
        if installation is not None:
            return installation.interface
        return INTERFACE_LUT["realitycapture.current"]

    def _find_realityscan(self) -> Path | None:
        installation = self._find_installation()
        return installation.executable if installation else None

    def _require_realityscan(self) -> Path:
        exe = self._find_realityscan()
        if exe is None:
            raise CapabilityUnavailableError(
                capability="backend.actions",
                reason=(
                    "RealityScan executable not found. Set SFMAPI_REALITYSCAN_EXECUTABLE "
                    "or SFMAPI_RC_EXECUTABLE, or pass --rc-executable to "
                    "sfmapi-realityscan-api."
                ),
            )
        return exe

    def _run(
        self, args: list[str], *, timeout_seconds: int | float | None = None
    ) -> subprocess.CompletedProcess[str]:
        exe = self._find_realityscan()
        env = os.environ.copy()
        if exe is not None:
            path_parts = [str(path) for path in realityscan_runtime_dirs(exe)]
            if path_parts:
                env["PATH"] = os.pathsep.join([*path_parts, env.get("PATH", "")])
        try:
            return subprocess.run(
                args,
                check=True,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout_seconds,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            raise ValidationError(f"RealityScan command failed: {detail}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ValidationError(
                f"RealityScan command timed out after {timeout_seconds}s"
            ) from exc

    def _command_action(self, command: str, *, include_schemas: bool) -> dict[str, Any]:
        interface = self._interface()
        descriptor = {
            "action_id": f"realityscan.{command}",
            "backend": self.name,
            "display_name": f"{interface.display_name} {command}",
            "description": f"Run the upstream {interface.display_name} `-{command}` command.",
            "category": _COMMAND_TO_CATEGORY.get(command, "utility"),
            "stability": "backend_extension",
            "side_effects": "write",
            "long_running": command not in {"getStatus", "printReport", "tag"},
            "supports_progress": True,
            "idempotent": command in {"getStatus", "printReport", "tag"},
            "gpu_required": command in _GPU_REQUIRED_COMMANDS,
            "required_capabilities": [],
            "metadata": {
                "family": "reality_cli",
                "product": interface.product,
                "interface_id": interface.interface_id,
                "command": command,
                "docs_url": interface.docs_url,
            },
        }
        if include_schemas:
            descriptor["input_schema"] = self._command_input_schema()
            descriptor["output_schema"] = self._run_output_schema()
        return descriptor

    def _run_sequence_action(self, *, include_schemas: bool) -> dict[str, Any]:
        interface = self._interface()
        descriptor = {
            "action_id": "realityscan.runSequence",
            "backend": self.name,
            "display_name": f"{interface.display_name} command sequence",
            "description": f"Run an ordered {interface.display_name} CLI command sequence.",
            "category": "utility",
            "stability": "backend_extension",
            "side_effects": "write",
            "long_running": True,
            "supports_progress": True,
            "idempotent": False,
            "gpu_required": True,
            "required_capabilities": [],
            "metadata": {
                "family": "reality_cli",
                "product": interface.product,
                "interface_id": interface.interface_id,
                "docs_url": interface.docs_url,
            },
        }
        if include_schemas:
            descriptor["input_schema"] = self._sequence_input_schema()
            descriptor["output_schema"] = self._run_output_schema()
        return descriptor

    def _reconstruct_action(self, *, include_schemas: bool) -> dict[str, Any]:
        interface = self._interface()
        descriptor = {
            "action_id": "realityscan.reconstructImageFolder",
            "backend": self.name,
            "display_name": f"{interface.display_name} reconstruct image folder",
            "description": (
                f"Create a {interface.display_name} project from an image folder and "
                "run alignment/model calculation."
            ),
            "category": "reconstruction",
            "stability": "backend_extension",
            "side_effects": "write",
            "long_running": True,
            "supports_progress": True,
            "idempotent": False,
            "gpu_required": True,
            "required_capabilities": [],
            "metadata": {
                "family": "reality_cli",
                "product": interface.product,
                "interface_id": interface.interface_id,
                "default_project_extension": interface.default_project_extension,
                "docs_url": interface.docs_url,
            },
        }
        if include_schemas:
            descriptor["input_schema"] = self._reconstruct_input_schema()
            descriptor["output_schema"] = self._run_output_schema()
        return descriptor

    def _workflow_action(
        self, workflow: dict[str, Any], *, include_schemas: bool
    ) -> dict[str, Any]:
        interface = self._interface()
        descriptor = {
            "action_id": workflow["action_id"],
            "backend": self.name,
            "display_name": f"{interface.display_name} {workflow['display_suffix']}",
            "description": str(workflow["description"]),
            "category": "reconstruction",
            "stability": "backend_extension",
            "side_effects": "write",
            "long_running": True,
            "supports_progress": True,
            "idempotent": False,
            "gpu_required": True,
            "required_capabilities": [],
            "metadata": {
                "family": "reality_cli",
                "product": interface.product,
                "interface_id": interface.interface_id,
                "default_project_extension": interface.default_project_extension,
                "workflow": workflow["method"],
                "docs_url": interface.docs_url,
            },
        }
        if include_schemas:
            descriptor["input_schema"] = self._workflow_input_schema(workflow)
            descriptor["output_schema"] = self._run_output_schema()
        return descriptor

    def _workflow_input_schema(self, workflow: dict[str, Any]) -> dict[str, Any]:
        base = self._common_run_properties()
        base["properties"].update(dict(workflow["extra_properties"]))
        base["properties"]["settings"] = {
            "type": "object",
            "additionalProperties": {"type": ["string", "number", "integer", "boolean"]},
        }
        base["required"] = list(workflow["required"])
        return base

    def _command_input_schema(self) -> dict[str, Any]:
        base = self._common_run_properties()
        base["properties"]["args"] = {
            "type": "array",
            "items": {"type": "string"},
            "description": "Arguments passed after this RealityScan command.",
        }
        return base

    def _sequence_input_schema(self) -> dict[str, Any]:
        base = self._common_run_properties()
        base["properties"]["commands"] = {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string", "enum": list(self._interface().commands)},
                    "args": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
        }
        base["required"] = ["commands"]
        return base

    def _reconstruct_input_schema(self) -> dict[str, Any]:
        base = self._common_run_properties()
        base["properties"].update(
            {
                "image_folder": {"type": "string"},
                "project_path": {"type": "string"},
                "quality": {"type": "string", "enum": ["preview", "normal", "high"]},
                "export_model_path": {"type": "string"},
                "export_params_path": {"type": "string"},
                "settings": {
                    "type": "object",
                    "additionalProperties": {"type": ["string", "number", "integer", "boolean"]},
                },
            }
        )
        base["required"] = ["image_folder", "project_path"]
        return base

    def _common_run_properties(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "headless": {"type": "boolean", "default": True},
                "std_console": {"type": "boolean", "default": True},
                "fail_on_error": {"type": "boolean", "default": True},
                "append_quit": {"type": "boolean", "default": True},
                "silent_crash_report_path": {"type": "string"},
                "write_progress_path": {"type": "string"},
                "progress_timeout_seconds": {"type": "number"},
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
                "progress_records": {"type": "array"},
            },
        }

    def _normalize_action_inputs(self, action_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
        self.get_backend_action(action_id)
        if action_id == "realityscan.runSequence":
            if "commands" not in inputs:
                raise ValidationError("commands is required")
            self._commands_from_inputs(inputs)
            return inputs
        if action_id == "realityscan.reconstructImageFolder":
            for field in ("image_folder", "project_path"):
                if not inputs.get(field):
                    raise ValidationError(f"{field} is required")
            quality = str(inputs.get("quality", "normal"))
            if quality not in {"preview", "normal", "high"}:
                raise ValidationError("quality must be one of: preview, normal, high")
            inputs["quality"] = quality
            return inputs
        if action_id in _WORKFLOW_ACTIONS_BY_ID:
            workflow = _WORKFLOW_ACTIONS_BY_ID[action_id]
            for field in workflow["required"]:
                if not inputs.get(field):
                    raise ValidationError(f"{field} is required")
            if "quality" in workflow["extra_properties"]:
                quality = str(inputs.get("quality", "normal"))
                if quality not in {"preview", "normal", "high"}:
                    raise ValidationError("quality must be one of: preview, normal, high")
                inputs["quality"] = quality
            if action_id == "realityscan.alignOnly" and not (
                inputs.get("export_registration_path")
                or inputs.get("export_sparse_point_cloud_path")
            ):
                raise ValidationError(
                    "align_only requires export_registration_path or export_sparse_point_cloud_path"
                )
            return inputs
        if action_id.startswith("realityscan."):
            command = self._command_from_action_id(action_id)
            self._validate_command(command)
            args = inputs.get("args", [])
            if args is None:
                inputs["args"] = []
            elif not isinstance(args, list):
                raise ValidationError("args must be an array of strings")
            else:
                inputs["args"] = [str(arg) for arg in args]
            return inputs
        raise NotFoundError(f"Backend action {action_id!r} not found")

    def _commands_from_inputs(self, inputs: dict[str, Any]) -> list[tuple[str, list[str]]]:
        raw_commands = inputs.get("commands")
        if not isinstance(raw_commands, list):
            raise ValidationError("commands must be an array")
        commands: list[tuple[str, list[str]]] = []
        for index, item in enumerate(raw_commands):
            if not isinstance(item, dict):
                raise ValidationError(f"commands[{index}] must be an object")
            command = str(item.get("name") or "")
            self._validate_command(command)
            args = item.get("args", [])
            if args is None:
                args = []
            if not isinstance(args, list):
                raise ValidationError(f"commands[{index}].args must be an array")
            commands.append((command, [str(arg) for arg in args]))
        return commands

    def _command_from_action_id(self, action_id: str) -> str:
        if not action_id.startswith("realityscan."):
            raise NotFoundError(f"Backend action {action_id!r} not found")
        command = action_id.removeprefix("realityscan.")
        self._validate_command(command)
        return command

    def _validate_command(self, command: str) -> None:
        interface = self._interface()
        aliases = interface.command_aliases or {}
        normalized = aliases.get(command, command)
        if normalized not in set(interface.commands):
            raise ValidationError(
                f"unknown {interface.display_name} command: {command!r} "
                f"for interface {interface.interface_id}"
            )

    def _read_progress_records(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                records.append(
                    {
                        "alg_id": int(parts[0]),
                        "progress": float(parts[1]),
                        "duration_seconds": float(parts[2]),
                        "estimated_remaining_seconds": float(parts[3]),
                        "event_type": parts[4].lstrip("#"),
                    }
                )
            except ValueError:
                continue
        return records

    def _emit_progress(self, progress: Any | None, records: list[dict[str, Any]]) -> None:
        if progress is None:
            return
        for record in records:
            try:
                progress.metric("realityscan.progress", float(record["progress"]))
            except Exception:
                return

    def _unsupported(self, capability: str, reason: str = "") -> CapabilityUnavailableError:
        return CapabilityUnavailableError(capability=capability, reason=reason)

    def _stringify(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)


__all__ = [
    "INTERFACE_LUT",
    "REALITYSCAN_COMMANDS",
    "REALITYSCAN_COMMAND_CATEGORIES",
    "REALITYSCAN_COMMAND_SET",
    "REALITY_CLI_ENV_VARS",
    "RealityCliInstallation",
    "RealityCliInterface",
    "RealityScanCliBackend",
    "configure_reality_cli_environment",
    "configure_realityscan_environment",
    "realityscan_runtime_dirs",
    "resolve_reality_cli_installation",
    "resolve_realityscan_executable",
]
