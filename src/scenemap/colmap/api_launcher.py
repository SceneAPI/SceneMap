"""Console launchers for the three COLMAP provider APIs.

One parser/configure/main pipeline parameterized per provider, replacing
the three near-copy ``api_launcher.py`` modules of the superseded repos.
Per-provider drift is preserved through :data:`PROVIDERS`:

- ``native`` (ex ``sfmapi-colmap``): four selectable backends
  (default ``colmap_cpp_native``), Nuitka-standalone bundled-COLMAP
  detection, no ``configure_colmap_environment`` hook (plain
  ``SFMAPI_COLMAP_EXECUTABLE`` resolution).
- ``pycolmap`` (ex ``sfmapi-pycolmap``): two selectable backends
  (default ``colmap_pycolmap``); COLMAP resolution via the pycolmap
  provider's ``configure_colmap_environment``.
- ``cli`` (ex ``sfmapi-colmap-cli``): pinned to ``colmap_cli``
  (environment cannot override, as before); COLMAP resolution via the
  cli provider's ``configure_colmap_environment``.

Unification notes (deliberate deltas from the sources, see README):
``--dry-run`` and ``--workspace-root`` are now available for every
provider (they were native-/pycolmap-only conveniences), and the
non-reload path launches uvicorn from the import string for every
provider (the native launcher used to pass the app object).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProviderProfile:
    prog: str
    description: str
    backend_choices: tuple[str, ...]
    default_backend: str
    server_import: str
    # Read SFMAPI_BACKEND as the --backend default (native/pycolmap did;
    # the cli launcher always forced colmap_cli).
    backend_from_env: bool
    # Nuitka-standalone bundled colmap.exe probing (native only).
    bundled_lookup: bool
    # Dotted path of the provider's configure_colmap_environment, or None
    # to resolve SFMAPI_COLMAP_EXECUTABLE directly (native behavior).
    configure_module: str | None


PROVIDERS: dict[str, ProviderProfile] = {
    "native": ProviderProfile(
        prog="sfmapi-colmap-api",
        description="Run sfmapi with the native COLMAP backend collection.",
        backend_choices=(
            "colmap_cpp_native",
            "colmap_cpp_inmemory",
            "colmap_pycolmap",
            "colmap_cli",
        ),
        default_backend="colmap_cpp_native",
        server_import="scenemap.colmap.native.server:app",
        backend_from_env=True,
        bundled_lookup=True,
        configure_module=None,
    ),
    "pycolmap": ProviderProfile(
        prog="sfmapi-pycolmap-api",
        description="Run sfmapi with the upstream PyCOLMAP backend.",
        backend_choices=("colmap_pycolmap", "colmap_cli"),
        default_backend="colmap_pycolmap",
        server_import="scenemap.colmap.pycolmap.server:app",
        backend_from_env=True,
        bundled_lookup=False,
        configure_module="scenemap.colmap.pycolmap.backend",
    ),
    "cli": ProviderProfile(
        prog="sfmapi-colmap-cli-api",
        description="Run sfmapi with the upstream COLMAP CLI backend.",
        backend_choices=("colmap_cli",),
        default_backend="colmap_cli",
        server_import="scenemap.colmap.cli.server:app",
        backend_from_env=False,
        bundled_lookup=False,
        configure_module="scenemap.colmap.cli.backend",
    ),
}


def build_parser(provider: str) -> argparse.ArgumentParser:
    profile = PROVIDERS[provider]
    parser = argparse.ArgumentParser(prog=profile.prog, description=profile.description)
    default_backend = profile.default_backend
    if profile.backend_from_env:
        default_backend = os.environ.get("SFMAPI_BACKEND", profile.default_backend)
    if len(profile.backend_choices) > 1:
        parser.add_argument(
            "--backend",
            default=default_backend,
            choices=profile.backend_choices,
            help="Backend registered by this package.",
        )
    else:
        parser.set_defaults(backend=profile.default_backend)
    parser.add_argument(
        "--colmap-executable",
        "--colmap",
        dest="colmap_executable",
        default=os.environ.get("SFMAPI_COLMAP_EXECUTABLE"),
        help="Optional path to colmap.exe, colmap, or the COLMAP install/build directory.",
    )
    parser.add_argument("--host", default=os.environ.get("SFMAPI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SFMAPI_PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload mode.")
    parser.add_argument("--workspace-root", default=os.environ.get("SFMAPI_WORKSPACE_ROOT"))
    parser.add_argument("--log-level", default=os.environ.get("SFMAPI_LOG_LEVEL", "info"))
    parser.add_argument(
        "--mcp",
        choices=("off", "local"),
        default=None,
        help="Set SFMAPI_MCP_MODE for this API process. Use 'local' to mount /mcp.",
    )
    parser.add_argument(
        "--mcp-mount-path",
        default=os.environ.get("SFMAPI_MCP_MOUNT_PATH"),
        help="Root-relative MCP mount path when --mcp local is used.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Configure environment and print the selected COLMAP executable without starting uvicorn.",
    )
    parser.set_defaults(provider=provider)
    return parser


def parse_args(argv: Sequence[str] | None = None, provider: str = "native") -> argparse.Namespace:
    return build_parser(provider).parse_args(argv)


def bundled_colmap_executable() -> Path | None:
    """Locate a COLMAP executable bundled next to a frozen launcher."""

    names = ("colmap.exe", "colmap.bat", "colmap") if os.name == "nt" else ("colmap",)
    roots = []
    if getattr(sys, "frozen", False) or "__compiled__" in globals():
        roots.append(Path(sys.executable).resolve().parent)
    roots.append(Path(__file__).resolve().parent)

    for root in roots:
        for candidate_root in (root / "bin", root):
            for name in names:
                candidate = candidate_root / name
                if candidate.exists():
                    return candidate
    return None


def _configure_colmap_hook(profile: ProviderProfile) -> Callable[..., Path | None]:
    import importlib

    module = importlib.import_module(profile.configure_module or "")
    return module.configure_colmap_environment


def configure_environment(args: argparse.Namespace) -> Path | None:
    profile = PROVIDERS[args.provider]
    os.environ.setdefault("SFMAPI_EPHEMERAL", "true")
    os.environ["SFMAPI_BACKEND"] = str(args.backend)
    os.environ.setdefault(
        "SFMAPI_DB_URL", "sqlite+aiosqlite:///file::memory:?cache=shared&uri=true"
    )
    os.environ.setdefault("SFMAPI_BLOB_BACKEND", "memory")
    os.environ.setdefault("SFMAPI_QUEUE_BACKEND", "inline")
    os.environ.setdefault("SFMAPI_INLINE_TASKS", "true")
    if args.mcp is not None:
        os.environ["SFMAPI_MCP_MODE"] = str(args.mcp)
    if args.mcp_mount_path:
        os.environ["SFMAPI_MCP_MOUNT_PATH"] = str(args.mcp_mount_path)
    if args.workspace_root:
        os.environ["SFMAPI_WORKSPACE_ROOT"] = str(Path(args.workspace_root).resolve())

    if profile.configure_module is None:
        colmap_executable = args.colmap_executable
        if not colmap_executable and profile.bundled_lookup:
            colmap_executable = bundled_colmap_executable()
        if colmap_executable:
            resolved = Path(colmap_executable).resolve()
            os.environ["SFMAPI_COLMAP_EXECUTABLE"] = str(resolved)
            return resolved
        return None

    configure_colmap_environment = _configure_colmap_hook(profile)
    return configure_colmap_environment(
        args.colmap_executable,
        validate=bool(args.colmap_executable or os.environ.get("SFMAPI_COLMAP_EXECUTABLE")),
    )


def main(argv: Sequence[str] | None = None, provider: str = "native") -> None:
    args = parse_args(argv, provider)
    colmap_executable = configure_environment(args)
    if args.dry_run:
        selected = str(colmap_executable) if colmap_executable else "PATH lookup at runtime"
        print(f"sfmapi backend: {args.backend}\nCOLMAP executable: {selected}")
        return

    import uvicorn

    uvicorn.run(
        PROVIDERS[provider].server_import,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


def main_native(argv: Sequence[str] | None = None) -> None:
    main(argv, provider="native")


def main_pycolmap(argv: Sequence[str] | None = None) -> None:
    main(argv, provider="pycolmap")


def main_cli(argv: Sequence[str] | None = None) -> None:
    main(argv, provider="cli")


if __name__ == "__main__":
    main()
