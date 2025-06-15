# ruff: noqa: S101

"""Tests for the training endpoint (/training/train) with invalid data."""

import uuid

import pytest
from fastapi import status
from fastapi.testclient import TestClient

LOCALTRAINMODEL_ID = "3d2f3e4b-8c7f-4d2a-9f1e-0a6f3e4d2a5b"


@pytest.mark.training
class TestTrainingInvalid:
    """Tests for the training endpoint (/training/train) with invalid data."""

    @staticmethod
    def test_invalid_model_id(client: TestClient) -> None:
        """Tests training with an invalid model_id and checks the error response."""
        payload = {
            "model_tag": "00000000-0000-0000-0000-000000000000",
            "dataset_ids": ["0b30b284-f7fe-4e6c-a270-17cafc5b5bcb"],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code in {404, 422}
        data = response.json()
        assert "model" in str(data).lower() or "not found" in str(data).lower()

    @staticmethod
    def test_invalid_dataset_id(client: TestClient) -> None:
        """Tests training with an invalid dataset_id and checks the error response."""
        payload = {
            "model_tag": LOCALTRAINMODEL_ID,
            "dataset_ids": ["00000000-0000-0000-0000-000000000000"],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code in {404, 422}
        data = response.json()
        assert "dataset" in str(data).lower() or "not found" in str(data).lower()

    @staticmethod
    def test_empty_dataset_list(client: TestClient) -> None:
        """Tests training with an empty dataset_ids list."""
        payload = {
            "model_id": LOCALTRAINMODEL_ID,
            "dataset_ids": [],
            "epochs": 3,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code in {422, 400}
        data = response.json()
        assert "dataset" in str(data).lower() or "empty" in str(data).lower()

    @staticmethod
    def test_negative_epochs(client: TestClient) -> None:
        """Tests training with a negative number of epochs (should fail validation)."""
        payload = {
            "model_id": LOCALTRAINMODEL_ID,
            "dataset_ids": ["0a9d5e87-e497-4737-9829-2070780d10df"],
            "epochs": -1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "epoch" in str(data).lower() or "positive" in str(data).lower()

    @staticmethod
    def test_zero_batch_size(client: TestClient) -> None:
        """Tests training with a zero batch size (should fail validation)."""
        payload = {
            "model_id": LOCALTRAINMODEL_ID,
            "dataset_ids": ["0a9d5e87-e497-4737-9829-2070780d10df"],
            "epochs": 3,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 0,
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "batch" in str(data).lower() or "zero" in str(data).lower()

    @staticmethod
    def test_negative_learning_rate(client: TestClient) -> None:
        """Tests training with a negative learning rate (should fail validation)."""
        payload = {
            "model_id": LOCALTRAINMODEL_ID,
            "dataset_ids": ["0a9d5e87-e497-4737-9829-2070780d10df"],
            "epochs": 3,
            "learning_rate": -0.01,
            "per_device_train_batch_size": 8,
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "learning" in str(data).lower() or "negative" in str(data).lower()

    @staticmethod
    def test_invalid_learning_rate_zero(client: TestClient) -> None:
        """Tests training with a learning rate of zero (should fail validation)."""
        payload = {
            "model_id": LOCALTRAINMODEL_ID,
            "dataset_ids": ["0a9d5e87-e497-4737-9829-2070780d10df"],
            "epochs": 3,
            "learning_rate": 0.0,
            "per_device_train_batch_size": 8,
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "learning" in str(data).lower() or "zero" in str(data).lower()

    @staticmethod
    def test_get_training_status_not_found(client: TestClient) -> None:
        """Tests the GET /training/{task_id}/status endpoint with an invalid task ID."""
        invalid_task_id = str(uuid.uuid4())
        response = client.get(f"/training/{invalid_task_id}/status")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "not found" in str(data).lower()
