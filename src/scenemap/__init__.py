"""scenemap: unified SfM mapping backend plugins for sceneapi.

One package, six ``sceneapi.backends`` entry points (names unchanged from
the four superseded repos):

- ``colmap_native`` -> :mod:`scenemap.colmap.native.plugin`
- ``pycolmap`` -> :mod:`scenemap.colmap.pycolmap.plugin`
- ``colmap_cli`` -> :mod:`scenemap.colmap.cli.plugin`
- ``instantsfm`` -> :mod:`scenemap.instantsfm.plugin`
- ``spheresfm`` -> :mod:`scenemap.spheresfm.plugin`
- ``realityscan_cli`` -> :mod:`scenemap.realityscan.plugin`

Supersedes the ``sfmapi_colmap_unified``, ``sfmapi_instantsfm``,
``sfmapi_spheresfm``, and ``sfmapi_realityscan`` repos (SceneAPI migration
W8; the COLMAP subpackage had itself absorbed ``sfmapi_colmap``,
``sfmapi_pycolmap``, and ``sfmapi_colmap_cli`` in the D3/L43 merge).

Kept import-free so loading one family's entry point does not import the
other three; import the family subpackages directly.
"""
