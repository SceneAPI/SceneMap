"""ASGI entrypoint that registers the COLMAP backend before sfmapi starts."""
# ruff: noqa: E402

from __future__ import annotations

import os

# Demo default: no persistent API database/blob state. Upstream COLMAP still
# needs filesystem paths for its SQLite database and model files, so sfmapi's
# ephemeral temp workspace acts as the scratch area and is removed on shutdown.
os.environ.setdefault("SFMAPI_EPHEMERAL", "true")
os.environ.setdefault("SFMAPI_BACKEND", "colmap_cli")
os.environ.setdefault("SFMAPI_DB_URL", "sqlite+aiosqlite:///file::memory:?cache=shared&uri=true")
os.environ.setdefault("SFMAPI_BLOB_BACKEND", "memory")
os.environ.setdefault("SFMAPI_QUEUE_BACKEND", "inline")
os.environ.setdefault("SFMAPI_INLINE_TASKS", "true")

from sceneapi_map.colmap.cli.backend import configure_colmap_environment
from sceneapi_map.colmap.cli.plugin import plugin

configure_colmap_environment(validate=bool(os.environ.get("SFMAPI_COLMAP_EXECUTABLE")))

from sceneapi.runtime import create_app, register_backend

plugin.register(register_backend)

app = create_app()
