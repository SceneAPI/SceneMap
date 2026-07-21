"""MapAnything feed-forward mapping provider for scenemap.

The learned-family proof point: :class:`scenemap.mapanything.backend.MapAnythingBackend`
implements the neutral :class:`sceneio.mapping.Mapper` contract with
``requires_correspondences=False``, so core's ``io_mapper()`` resolver routes
feed-forward mapping to it with no core routing changes.

Kept import-free (no torch / mapanything at import time) so loading this
provider's entry point does not pull the heavy engine; import
:mod:`scenemap.mapanything.backend` / ``.plugin`` directly.
"""
