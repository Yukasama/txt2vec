"""Test cases for upload status endpoint with valid/invalid inputs."""
# ruff: noqa: S101

import pytest
from fastapi import status
from fastapi.testclient import TestClient

UPLOAD_TASK_GH_ID = "d2f3e4b8-8c7f-4d2a-9f1e-0a6f3e4d2a5b"
UPLOAD_TASK_HF_ID = "d2f3e4b8-8c7f-4d2a-9f1e-0a6f3e4d2a5c"


@pytest.mark.upload
@pytest.mark.parametrize("task_id", [
    UPLOAD_TASK_GH_ID,
    UPLOAD_TASK_HF_ID
])
def test_valid_input_returns_200(client: TestClient, task_id: str) -> None:
    """Existing task ID should return 200 OK with task data."""
    response = client.get(f"/uploads/{task_id}")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["id"] == task_id


@pytest.mark.upload
@pytest.mark.parametrize("task_id", [
    "d84eadb2-eddb-462d-989b-34578c5dd164",
    "00000000-0000-0000-0000-000000000000"
])
def test_valid_uuid_not_found_returns_404(client: TestClient, task_id: str) -> None:
    """Valid UUIDs that do not match any task should return 404."""
    response = client.get(f"/uploads/{task_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.upload
@pytest.mark.parametrize("task_id", [
    "invalid-id",
    "abc-xyz",
    "12345678-1234-5678-1234-5678123456789",
    "12345678-1234",
    "!@#$%^&*()",
])
def test_invalid_uuid_returns_422(client: TestClient, task_id: str) -> None:
    """Malformed UUIDs should trigger FastAPI 422 validation error."""
    response = client.get(f"/uploads/{task_id}")
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
