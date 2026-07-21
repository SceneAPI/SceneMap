"""Runtime-version info commands for the three providers.

Replaces the per-repo ``cli.py`` modules (renamed: a top-level ``cli``
module would collide with the :mod:`scenemap.colmap.cli` provider
package). Each ``main_*`` prints exactly what the superseded repo's
``sfmapi-*-info`` command printed.
"""

from __future__ import annotations

import json


def main_native() -> None:
    from .native.backend import ColmapCliBackend
    from .native.cpp_inmemory_backend import CppInmemoryBackend
    from .native.cpp_native_backend import CppNativeBackend
    from .pycolmap_backend import PycolmapBackend

    backends = {
        "colmap_pycolmap": PycolmapBackend().runtime_versions(),
        "colmap_cpp_native": CppNativeBackend().runtime_versions(),
        "colmap_cpp_inmemory": CppInmemoryBackend().runtime_versions(),
        "colmap_cli": ColmapCliBackend().runtime_versions(),
    }
    print(json.dumps(backends, indent=2, sort_keys=True))


def main_pycolmap() -> None:
    from .pycolmap.backend import ColmapCliBackend
    from .pycolmap_backend import PycolmapBackend

    backends = {
        "colmap_pycolmap": PycolmapBackend().runtime_versions(),
        "colmap_cli": ColmapCliBackend().runtime_versions(),
    }
    print(json.dumps(backends, indent=2, sort_keys=True))


def main_cli() -> None:
    from .cli.backend import ColmapCliBackend

    backend = ColmapCliBackend()
    print(json.dumps(backend.runtime_versions(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main_native()
