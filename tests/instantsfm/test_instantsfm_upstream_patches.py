from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest


def test_tracks_accept_colmap_style_global_track_ids() -> None:
    instantsfm_root = Path(__file__).resolve().parents[2] / "third_party" / "instantsfm"
    # Merge adaptation: this repo registers the third_party/instantsfm
    # gitlink without checking its contents out (same convention as the
    # colmap and spheresfm submodules); the source repo always ran against
    # an initialized submodule. Skip until it is materialized. NOTE: with
    # the submodule initialized this test asserts the upstream int64
    # track-id patch is present in the *working tree* -- a pristine
    # checkout of the pinned commit passes, but a stale local working
    # tree (as in the superseded sfmapi_instantsfm checkout) fails until
    # provisioning re-applies `_patch_instantsfm_source`.
    if not (instantsfm_root / "instantsfm").is_dir():
        pytest.skip(
            "third_party/instantsfm not initialized; run "
            "`git submodule update --init third_party/instantsfm`"
        )
    sys.path.insert(0, str(instantsfm_root))
    sys.modules.setdefault("cv2", types.SimpleNamespace())
    from instantsfm.scene.defs import Tracks

    tracks = Tracks(num_tracks=1)
    global_track_id = (25 << 32) | 12345

    tracks.ids[0] = global_track_id

    assert tracks.ids.dtype == np.int64
    assert int(tracks.ids[0]) == global_track_id
