"""MapAnything feed-forward mapping provider for sceneapi-map.

The learned-family proof point: :class:`sceneapi_map.mapanything.backend.MapAnythingBackend`
implements the neutral :class:`sceneapi_io.mapping.Mapper` contract with
``requires_correspondences=False``, so core's ``io_mapper()`` resolver routes
feed-forward mapping to it with no core routing changes.

Kept import-free (no torch / mapanything at import time) so loading this
provider's entry point does not pull the heavy engine; import
:mod:`sceneapi_map.mapanything.backend` / ``.plugin`` directly.
"""
