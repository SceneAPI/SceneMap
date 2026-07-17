from __future__ import annotations

import importlib
import subprocess
from pathlib import Path
from typing import Any

from .backend import (
    CapabilityUnavailableError,
    ColmapCliBackend,
    ValidationError,
    colmap_runtime_path_dirs,
)


class CppNativeBackend(ColmapCliBackend):
    """COLMAP backend that routes native commands through the C++ extension."""

    name = "colmap_cpp_native"
    version = "0.0.1"
    vendor = "COLMAP upstream / sfmapi-colmap C++ bridge"

    def capabilities(self) -> set[str]:
        try:
            self._require_cpp("capabilities")
        except CapabilityUnavailableError:
            return set()
        capabilities = super().capabilities()
        if not capabilities:
            return set()
        return capabilities

    def runtime_versions(self) -> dict[str, str]:
        versions = super().runtime_versions()
        try:
            cpp = self._require_cpp("runtime.cpp")
        except CapabilityUnavailableError:
            versions["cpp_native"] = "missing"
        else:
            versions["cpp_native"] = str(cpp.version())
        return versions

    def _run(
        self,
        args: list[str],
        *,
        progress: Any | None = None,
        progress_phase: str | None = None,
        progress_total: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        cpp = self._require_cpp("colmap.native")
        executable = Path(args[0]) if args else None
        path_prefix = [str(path) for path in colmap_runtime_path_dirs(executable)]
        normalized_args = [str(arg) for arg in args]
        try:
            raw = cpp.run_command(normalized_args, path_prefix)
        except Exception as exc:
            command = " ".join(normalized_args[:2]) if normalized_args else "colmap"
            raise ValidationError(f"{command} failed to start through C++ bridge: {exc}") from exc

        result = subprocess.CompletedProcess(
            normalized_args,
            int(raw["returncode"]),
            str(raw.get("stdout", "")),
            str(raw.get("stderr", "")),
        )
        if result.returncode != 0:
            command = " ".join(normalized_args[:2])
            detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
            raise ValidationError(f"{command} failed: {detail}") from None
        if progress is not None and progress_phase is not None and progress_total is not None:
            self._progress(progress, progress_phase, current=progress_total, total=progress_total)
        return result

    def _colmap_help_header(self, exe: Path) -> str:
        try:
            result = self._run([str(exe), "-h"])
        except Exception:
            return "unknown"
        for line in (result.stdout + result.stderr).splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:160]
        return "unknown"

    def _require_cpp(self, capability: str) -> Any:
        try:
            return importlib.import_module("sfmapi_colmap._cpp_inmemory")
        except (ImportError, RuntimeError) as exc:
            raise CapabilityUnavailableError(
                capability=capability,
                reason=(
                    "the sfmapi_colmap._cpp_inmemory extension is not installed; "
                    "sceneapi-map does not build it — install a wheel built "
                    "from the superseded sfmapi_colmap repo (scikit-build-core) to "
                    "enable the C++ demo providers"
                ),
            ) from exc


__all__ = ["CppNativeBackend"]
