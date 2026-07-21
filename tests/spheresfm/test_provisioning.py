from __future__ import annotations

import pytest

from scenemap.spheresfm import provisioning


def _asset(name: str, url: str = "https://example.test/asset.zip") -> dict[str, str]:
    return {"name": name, "browser_download_url": url}


def test_release_asset_selection_accepts_spheresfm_zip() -> None:
    release = {"assets": [_asset("notes.txt"), _asset("SphereSfM-windows.zip")]}

    selected = provisioning._select_release_asset(release)

    assert selected["name"] == "SphereSfM-windows.zip"


def test_release_asset_selection_rejects_generic_zip() -> None:
    with pytest.raises(RuntimeError, match="downloadable zip asset"):
        provisioning._select_release_asset({"assets": [_asset("generic-windows.zip")]})


def test_release_asset_selection_rejects_missing_download() -> None:
    with pytest.raises(RuntimeError, match="downloadable zip asset"):
        provisioning._select_release_asset({"assets": [_asset("SphereSfM-windows.zip", url="")]})
