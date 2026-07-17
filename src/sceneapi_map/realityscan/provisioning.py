"""Install-time runtime detection for the RealityScan CLI backend."""

from __future__ import annotations

from typing import Any

from .backend import configure_reality_cli_environment


def _step(name: str, action: str, status: str, **extra: object) -> dict[str, object]:
    return {"name": name, "action": action, "status": status, **extra}


def provision(*, dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    if dry_run:
        return {
            "available": True,
            "provisioned": False,
            "steps": [
                _step(
                    "realityscan_detect",
                    "detect an installed RealityCapture.exe or RealityScan.exe",
                    "planned",
                )
            ],
            "env": {},
            "warnings": [
                "RealityCapture/RealityScan is proprietary and is not downloaded by sfmapi."
            ],
        }

    installation = configure_reality_cli_environment(validate=False)
    if installation is None:
        return {
            "available": True,
            "provisioned": False,
            "steps": [
                _step(
                    "realityscan_detect",
                    "detect an installed RealityCapture.exe or RealityScan.exe",
                    "failed",
                )
            ],
            "env": {},
            "warnings": [
                "RealityCapture/RealityScan is proprietary and was not found. Install it "
                "from Epic/Capturing Reality, then set SFMAPI_RC_EXECUTABLE or "
                "SFMAPI_REALITYSCAN_EXECUTABLE."
            ],
        }

    return {
        "available": True,
        "provisioned": True,
        "steps": [
            _step(
                "realityscan_detect",
                "detect an installed RealityCapture.exe or RealityScan.exe",
                "done",
                path=str(installation.executable),
            )
        ],
        "env": {"SFMAPI_RC_EXECUTABLE": str(installation.executable)},
        "warnings": [],
    }
