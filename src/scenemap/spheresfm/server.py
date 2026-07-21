"""ASGI entrypoint that registers the SphereSfM backend before sfmapi starts."""
# ruff: noqa: E402

from __future__ import annotations

import os

os.environ.setdefault("SFMAPI_EPHEMERAL", "true")
os.environ.setdefault("SFMAPI_BACKEND", "spheresfm")
os.environ.setdefault("SFMAPI_DB_URL", "sqlite+aiosqlite:///file::memory:?cache=shared&uri=true")
os.environ.setdefault("SFMAPI_BLOB_BACKEND", "memory")
os.environ.setdefault("SFMAPI_QUEUE_BACKEND", "inline")
os.environ.setdefault("SFMAPI_INLINE_TASKS", "true")

from scenemap.spheresfm.backend import configure_spheresfm_environment
from scenemap.spheresfm.plugin import plugin

configure_spheresfm_environment(validate=bool(os.environ.get("SFMAPI_SPHERESFM_EXECUTABLE")))

from sceneapi.runtime import create_app, register_backend

plugin.register(register_backend)

app = create_app()
