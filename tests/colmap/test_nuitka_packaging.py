from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_api_launcher_console_script_is_registered():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["sfmapi-colmap-api"] == (
        "sceneapi_map.colmap.api_launcher:main_native"
    )
    assert pyproject["project"]["entry-points"]["sceneapi.backends"]["colmap_native"] == (
        "sceneapi_map.colmap.native.plugin:plugin"
    )
    assert "nuitka>=2.6" in pyproject["project"]["optional-dependencies"]["standalone"]
    assert "fastmcp==3.2.4" in pyproject["project"]["optional-dependencies"]["mcp"]
    assert "fastmcp==3.2.4" in pyproject["project"]["optional-dependencies"]["standalone"]


def test_nuitka_standalone_scripts_include_api_and_runtime_inputs():
    scripts = [
        (REPO_ROOT / "scripts" / "build-nuitka-standalone.ps1").read_text(encoding="utf-8"),
        (REPO_ROOT / "scripts" / "build-nuitka-standalone.sh").read_text(encoding="utf-8"),
    ]

    for text in scripts:
        assert "--standalone" in text
        # Merge adaptation: the core's `app` shim is gone in sceneapi 0.1.0;
        # the frozen build bundles the renamed core package instead.
        assert "--include-package=sceneapi" in text
        assert "--include-package=sceneapi_map.colmap" in text
        assert "--include-package=uvicorn" in text
        assert "--include-package=fastmcp" in text
        assert "SFMAPI_MCP_MODE=off" in text
        assert "src/sceneapi_map/colmap/api_launcher.py" in text
        assert "colmap_cpp_native" in text
        assert "colmap-install-cuda-cudss" in text
