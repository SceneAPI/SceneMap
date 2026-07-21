"""pycolmap provider package (ex ``sfmapi_pycolmap``).

``backend.py`` is the base ``ColmapCliBackend`` variant the reconciled
:mod:`scenemap.colmap.pycolmap_backend` builds on. Kept
import-free to avoid plugin/backend import cycles; import the concrete
modules directly.
"""
