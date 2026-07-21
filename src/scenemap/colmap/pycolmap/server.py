"""ASGI entrypoint that registers the COLMAP backend before sfmapi starts."""
# ruff: noqa: I001

from __future__ import annotations

import os

# Demo default: no persistent API database/blob state. PyCOLMAP still uses
# COLMAP database/model files for SfM work, so sfmapi's ephemeral temp
# workspace acts as the scratch area and is removed on shutdown.
os.environ.setdefault("SFMAPI_EPHEMERAL", "true")
os.environ.setdefault("SFMAPI_BACKEND", "colmap_pycolmap")
os.environ.setdefault("SFMAPI_DB_URL", "sqlite+aiosqlite:///file::memory:?cache=shared&uri=true")
os.environ.setdefault("SFMAPI_BLOB_BACKEND", "memory")
os.environ.setdefault("SFMAPI_QUEUE_BACKEND", "inline")
os.environ.setdefault("SFMAPI_INLINE_TASKS", "true")

from sceneapi.runtime import create_app, register_backend
from scenemap.colmap.pycolmap.plugin import plugin

plugin.register(register_backend)

app = create_app()
