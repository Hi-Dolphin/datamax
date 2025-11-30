from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

TEST_DB_PATH = Path("test_operations.db")
if TEST_DB_PATH.exists():
    TEST_DB_PATH.unlink()

os.environ.setdefault("DATAMAX_POSTGRES_DSN", "sqlite+aiosqlite:///./test_operations.db")
os.environ.setdefault("DATAMAX_ENVIRONMENT", "test")

from datamax.operations.main import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_datasources(client: TestClient) -> None:
    response = client.get("/api/v1/datasources")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert payload, "seeded mock store should return at least one datasource"
    first = payload[0]
    assert "name" in first
    assert "connection" in first


def test_create_dataset(client: TestClient) -> None:
    datasources = client.get("/api/v1/datasources").json()
    datasource_id = datasources[0]["id"]

    payload = {
        "name": "测试数据集",
        "datasource_id": datasource_id,
        "description": "用于单元测试的临时数据集",
        "format": "parquet",
        "record_count": 123,
        "tags": ["unit", "tmp"],
    }
    response = client.post("/api/v1/datasets", json=payload)
    assert response.status_code == 201, response.text
    data = response.json()
    assert data["name"] == payload["name"]
    assert data["datasource_id"] == datasource_id


def test_pipeline_metrics_available(client: TestClient) -> None:
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    metrics = response.json()
    assert isinstance(metrics, list)
    assert metrics, "mock metrics should be returned"
    for summary in metrics:
        assert "scope" in summary
        assert "totals" in summary


def test_alert_acknowledge(client: TestClient) -> None:
    alerts_resp = client.get("/api/v1/alerts")
    assert alerts_resp.status_code == 200
    alerts = alerts_resp.json()
    assert alerts, "mock store seeds one alert"
    alert_id = alerts[0]["id"]

    ack_resp = client.post(f"/api/v1/alerts/{alert_id}/ack")
    assert ack_resp.status_code == 200
    acknowledged = ack_resp.json()
    assert acknowledged["acknowledged"] is True


def test_dataset_refresh_updates_timestamp(client: TestClient) -> None:
    list_resp = client.get("/api/v1/datasets")
    assert list_resp.status_code == 200
    dataset = list_resp.json()[0]
    dataset_id = dataset["id"]
    original_timestamp = dataset.get("last_refreshed_at")

    refresh_resp = client.post(f"/api/v1/datasets/{dataset_id}/refresh")
    assert refresh_resp.status_code == 200
    refreshed = refresh_resp.json()

    assert refreshed["record_count"] >= dataset["record_count"]
    assert refreshed["last_refreshed_at"] != original_timestamp


def teardown_module() -> None:
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
