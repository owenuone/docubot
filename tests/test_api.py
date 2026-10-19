import pytest
from fastapi.testclient import TestClient

from app.main import app

HEADERS = {"X-API-Key": "test-token"}


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def test_health_is_public(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_api_requires_key(client: TestClient) -> None:
    assert client.get("/api/documents", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/documents").status_code == 401


def test_upload_ask_delete_flow(client: TestClient) -> None:
    files = {"file": ("shipping.md", b"We ship worldwide. Delivery takes 5 to 7 business days.")}
    resp = client.post("/api/documents", files=files, headers=HEADERS)
    assert resp.status_code == 201
    doc_id = resp.json()["id"]

    resp = client.post("/api/ask", json={"question": "How long does delivery take?"}, headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json()["sources"][0]["label"] == "shipping.md"

    logs = client.get("/api/logs", headers=HEADERS).json()
    assert logs[0]["question"] == "How long does delivery take?"

    assert client.delete(f"/api/documents/{doc_id}", headers=HEADERS).status_code == 204
    assert all(d["id"] != doc_id for d in client.get("/api/documents", headers=HEADERS).json())


def test_upload_unsupported_type(client: TestClient) -> None:
    files = {"file": ("sheet.xlsx", b"binary")}
    assert client.post("/api/documents", files=files, headers=HEADERS).status_code == 422
