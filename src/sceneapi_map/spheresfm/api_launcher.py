from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from .backend import configure_spheresfm_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sfmapi-spheresfm-api",
        description="Run sfmapi with the upstream SphereSfM action backend.",
    )
    parser.add_argument(
        "--spheresfm-executable",
        "--colmap",
        dest="spheresfm_executable",
        default=os.environ.get("SFMAPI_SPHERESFM_EXECUTABLE"),
        help="Path to the SphereSfM-built colmap executable.",
    )
    parser.add_argument("--host", default=os.environ.get("SFMAPI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SFMAPI_PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload mode.")
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
        help="Configure environment and print the selected SphereSfM executable.",
    )
    return parser


def configure_environment(args: argparse.Namespace) -> Path | None:
    os.environ.setdefault("SFMAPI_EPHEMERAL", "true")
    os.environ["SFMAPI_BACKEND"] = "spheresfm"
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
    return configure_spheresfm_environment(
        args.spheresfm_executable,
        validate=bool(args.spheresfm_executable or os.environ.get("SFMAPI_SPHERESFM_EXECUTABLE")),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    executable = configure_environment(args)
    if args.dry_run:
        selected = str(executable) if executable else "missing until built/configured"
        print(f"sfmapi backend: spheresfm\nSphereSfM executable: {selected}")
        return

    import uvicorn

    uvicorn.run(
        "sceneapi_map.spheresfm.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
