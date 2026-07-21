from __future__ import annotations

import argparse
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from .backend import InstantSfMBackend
from .plugin import manifest

PROTOCOL = "sfmapi-plugin-http-v1"
PROTOCOL_VERSION = "1.0"


def _package_version() -> str:
    try:
        return version("scenemap")
    except PackageNotFoundError:
        return "0.0.0+editable"


backend = InstantSfMBackend()
app = FastAPI(title="sfmapi InstantSfM plugin service")


def _workspace_from_payload(payload: dict[str, Any]) -> Path:
    mounts = payload.get("mounts") if isinstance(payload.get("mounts"), dict) else {}
    work = mounts.get("work") if isinstance(mounts.get("work"), dict) else {}
    host_path = work.get("host_path") or work.get("container_path")
    path = Path(str(host_path or os.environ.get("SFMAPI_PLUGIN_WORKDIR", "/sfmapi/work")))
    path.mkdir(parents=True, exist_ok=True)
    return path


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
def service_version() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "protocol_version": PROTOCOL_VERSION,
        "plugin_id": manifest["plugin_id"],
        "package_version": _package_version(),
        "backend": backend.runtime_versions(),
    }


@app.get("/capabilities")
def capabilities() -> dict[str, Any]:
    return {"capabilities": sorted(backend.capabilities())}


@app.get("/actions")
def actions() -> dict[str, Any]:
    return {"items": backend.list_backend_actions(include_schemas=True)}


@app.post("/actions/{action_id}:validate")
def validate_action(action_id: str, body: dict[str, Any]) -> dict[str, Any]:
    return backend.validate_backend_action(action_id, dict(body or {}))


@app.post("/actions/{action_id}:run")
def run_action(action_id: str, body: dict[str, Any]) -> dict[str, Any]:
    try:
        result = backend.run_backend_action(action_id, dict(body or {}))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "succeeded", "outputs": result}


@app.post("/execute")
def execute(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("protocol") not in (None, PROTOCOL):
        return {
            "status": "failed",
            "error": {
                "code": "protocol_mismatch",
                "message": f"expected {PROTOCOL}",
                "target": "/execute",
            },
            "outputs": {},
        }
    action_id = str(payload.get("action_id") or "")
    if not action_id:
        return {
            "status": "failed",
            "error": {
                "code": "missing_action_id",
                "message": "action_id is required",
                "target": "/execute",
            },
            "outputs": {},
        }
    try:
        result = backend.run_backend_action(
            action_id,
            dict(payload.get("inputs") or {}),
            workspace=_workspace_from_payload(payload),
        )
    except Exception as exc:
        return {
            "status": "failed",
            "error": {
                "code": type(exc).__name__,
                "message": str(exc),
                "target": "/execute",
            },
            "outputs": {},
            "artifacts": [],
            "logs": {},
        }
    return {"status": "succeeded", "outputs": result, "artifacts": [], "logs": {}}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        default=os.environ.get("SFMAPI_INSTANTSFM_SERVICE_HOST", "0.0.0.0"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SFMAPI_INSTANTSFM_SERVICE_PORT", "8096")),
    )
    args = parser.parse_args(argv)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
