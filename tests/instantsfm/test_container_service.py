from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from scenemap.instantsfm import container_service


class _FakeBackend:
    def runtime_versions(self) -> dict[str, str]:
        return {"backend": "test"}

    def capabilities(self) -> set[str]:
        return {"map.global"}

    def list_backend_actions(self, *, include_schemas: bool = False) -> list[dict[str, object]]:
        return [{"action_id": "instantsfm.runPipeline", "schemas": include_schemas}]

    def validate_backend_action(
        self, action_id: str, inputs: dict[str, object]
    ) -> dict[str, object]:
        return {"action_id": action_id, "valid": True, "normalized_inputs": inputs, "errors": []}

    def run_backend_action(
        self,
        action_id: str,
        inputs: dict[str, object],
        *,
        workspace: Path | None = None,
    ) -> dict[str, object]:
        assert workspace is not None
        return {"action_id": action_id, "inputs": inputs, "workspace": str(workspace)}


def test_container_service_health_version_actions_and_execute(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(container_service, "backend", _FakeBackend())
    client = TestClient(container_service.app)

    assert client.get("/healthz").json() == {"status": "ok"}
    version = client.get("/version").json()
    assert version["protocol"] == container_service.PROTOCOL
    assert version["plugin_id"] == "instantsfm"
    assert client.get("/capabilities").json() == {"capabilities": ["map.global"]}
    assert client.get("/actions").json()["items"][0]["action_id"] == "instantsfm.runPipeline"

    response = client.post(
        "/execute",
        json={
            "protocol": container_service.PROTOCOL,
            "action_id": "instantsfm.runPipeline",
            "inputs": {"data_path": "data"},
            "mounts": {"work": {"host_path": str(tmp_path / "work")}},
        },
    )

    body = response.json()
    assert body["status"] == "succeeded"
    assert body["outputs"]["action_id"] == "instantsfm.runPipeline"
    assert body["artifacts"] == []


def test_container_service_execute_reports_contract_errors() -> None:
    client = TestClient(container_service.app)

    bad_protocol = client.post("/execute", json={"protocol": "wrong"}).json()
    assert bad_protocol["status"] == "failed"
    assert bad_protocol["error"]["code"] == "protocol_mismatch"

    missing_action = client.post("/execute", json={}).json()
    assert missing_action["status"] == "failed"
    assert missing_action["error"]["code"] == "missing_action_id"
