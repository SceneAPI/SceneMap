"""Unified COLMAP backend plugins for sfmapi.

One package, three ``sceneapi.backends`` entry points (unchanged names):

- ``colmap_native`` -> :mod:`scenemap.colmap.native.plugin` (four
  provider ids: colmap_cli, colmap_pycolmap, colmap_cpp_native,
  colmap_cpp_inmemory)
- ``pycolmap`` -> :mod:`scenemap.colmap.pycolmap.plugin`
- ``colmap_cli`` -> :mod:`scenemap.colmap.cli.plugin`

Supersedes the ``sfmapi_colmap``, ``sfmapi_pycolmap``, and
``sfmapi_colmap_cli`` repos (lean-audit item 4.4, decision D3/L43).
"""

from __future__ import annotations

# Import order matters: native.plugin pulls ..pycolmap_backend, which binds to
# .pycolmap.backend — keep provider __init__ modules import-free so this chain
# stays acyclic.
from .cli.plugin import plugin as colmap_cli_plugin
from .native.plugin import plugin as colmap_native_plugin
from .pycolmap.plugin import plugin as pycolmap_plugin
from .pycolmap_backend import PycolmapBackend

__all__ = [
    "PycolmapBackend",
    "colmap_cli_plugin",
    "colmap_native_plugin",
    "pycolmap_plugin",
]
