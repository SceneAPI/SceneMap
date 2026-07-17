from __future__ import annotations

import pytest

from sceneapi_map.colmap import provisioning

# The release-asset selection tests shipped byte-identical (modulo
# imports) in sfmapi_colmap and sfmapi_colmap_cli; the machinery is now
# shared, so they run once.


def _asset(name: str, url: str = "https://example.test/asset.zip") -> dict[str, str]:
    return {"name": name, "browser_download_url": url}


def test_release_asset_selection_prefers_requested_flavor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFMAPI_COLMAP_RELEASE_FLAVOR", "cuda")
    release = {
        "assets": [
            _asset("COLMAP-3.12-windows-nocuda.zip"),
            _asset("COLMAP-3.12-windows-cuda.zip"),
        ]
    }

    selected = provisioning._select_release_asset(release)

    assert selected["name"] == "COLMAP-3.12-windows-cuda.zip"


def test_release_asset_selection_falls_back_to_windows_zip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SFMAPI_COLMAP_RELEASE_FLAVOR", "missing")
    release = {"assets": [_asset("COLMAP-3.12-linux.zip"), _asset("COLMAP-3.12-windows.zip")]}

    selected = provisioning._select_release_asset(release)

    assert selected["name"] == "COLMAP-3.12-windows.zip"


def test_release_asset_selection_rejects_missing_download() -> None:
    with pytest.raises(RuntimeError, match="Windows zip asset"):
        provisioning._select_release_asset(
            {"assets": [_asset("COLMAP-3.12-windows.zip", url=""), _asset("README.txt")]}
        )


# Pin the per-provider provisioning surface of the merged module: each
# provision_* dry-run must keep planning exactly what its source repo
# planned (native = executable + pycolmap wheel; pycolmap = wheel only;
# cli = executable only), and the package-level sfm_hub hook must plan
# the superset.
@pytest.mark.parametrize(
    ("provision_fn", "expected_steps"),
    [
        pytest.param(
            provisioning.provision_native,
            ["colmap_runtime", "pycolmap_wheel"],
            id="native",
        ),
        pytest.param(provisioning.provision_pycolmap, ["pycolmap_wheel"], id="pycolmap"),
        pytest.param(provisioning.provision_cli, ["colmap_runtime"], id="cli"),
        pytest.param(provisioning.provision, ["colmap_runtime", "pycolmap_wheel"], id="hub-hook"),
    ],
)
def test_dry_run_plans_provider_specific_steps(provision_fn, expected_steps) -> None:
    result = provision_fn(dry_run=True)

    assert result["available"] is True
    assert result["provisioned"] is False
    assert [step["name"] for step in result["steps"]] == expected_steps
    assert all(step["status"] == "planned" for step in result["steps"])
