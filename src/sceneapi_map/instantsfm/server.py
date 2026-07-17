"""ASGI entrypoint that registers the InstantSfM backend before sfmapi starts."""
# ruff: noqa: E402

from __future__ import annotations

import os

os.environ.setdefault("SFMAPI_EPHEMERAL", "true")
os.environ.setdefault("SFMAPI_BACKEND", "instantsfm")
os.environ.setdefault("SFMAPI_DB_URL", "sqlite+aiosqlite:///file::memory:?cache=shared&uri=true")
os.environ.setdefault("SFMAPI_BLOB_BACKEND", "memory")
os.environ.setdefault("SFMAPI_QUEUE_BACKEND", "inline")
os.environ.setdefault("SFMAPI_INLINE_TASKS", "true")

from sceneapi_map.instantsfm.backend import configure_instantsfm_environment
from sceneapi_map.instantsfm.plugin import plugin

configure_instantsfm_environment(validate=bool(os.environ.get("SFMAPI_INSTANTSFM_ROOT")))

from sceneapi.runtime import create_app, register_backend

plugin.register(register_backend)

app = create_app()
