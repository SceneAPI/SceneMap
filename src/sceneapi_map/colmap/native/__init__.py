"""colmap_native provider package (ex ``sfmapi_colmap``).

Carries the native plugin's four backends: the COLMAP CLI wrapper, the
pycolmap-based backend (shared, reconciled — see
:mod:`sceneapi_map.colmap.pycolmap_backend`), and the two C++ demo
backends. Kept import-free to avoid plugin/backend import cycles;
import the concrete modules directly.
"""
