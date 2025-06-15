# ruff: noqa: S101

"""Tests for the training endpoint (/training/train) with valid data."""

import re
import shutil
import uuid
from pathlib import Path

import pytest
from fastapi import status
from fastapi.testclient import TestClient

MINILM_MODEL_TAG = "models--sentence-transformers--all-MiniLM-L6-v2"
DATASET_ID_1 = "0b30b284-f7fe-4e6c-a270-17cafc5b5bcb"
DATASET_ID_2 = "0a9d5e87-e497-4737-9829-2070780d10df"
DEFAULT_EPOCHS = 3
DEFAULT_LR = 0.00005
DEFAULT_BATCH_SIZE = 8
TRAINED_MODELS_DIR = Path("data/models/trained_models")

HTTP_200_OK = status.HTTP_200_OK
HTTP_202_ACCEPTED = status.HTTP_202_ACCEPTED
HTTP_404_NOT_FOUND = status.HTTP_404_NOT_FOUND


def ensure_minilm_model_available() -> None:
    """Ensure the required model files are present in data/models for training tests."""
    src = Path("test_data/training/models--sentence-transformers--all-MiniLM-L6-v2")
    dst = Path("data/models/models--sentence-transformers--all-MiniLM-L6-v2")
    if not dst.exists() and src.exists():
        shutil.copytree(src, dst)


@pytest.mark.training
class TestTrainingValid:
    """Tests for the training endpoint (/training/train) with valid data."""

    @staticmethod
    def test_valid_training(client: TestClient) -> None:
        """Test training with valid data and check response and status tracking."""
        ensure_minilm_model_available()
        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [DATASET_ID_1],
            "val_dataset_id": DATASET_ID_2,
            "epochs": DEFAULT_EPOCHS,
            "learning_rate": DEFAULT_LR,
            "per_device_train_batch_size": DEFAULT_BATCH_SIZE,
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED
        task_id = None
        if response.headers.get("Location"):
            match = re.search(
                r"/training/([a-f0-9\-]+)/status",
                response.headers["Location"],
            )
            if match:
                task_id = match.group(1)
        elif response.content and response.headers.get(
            "content-type", ""
        ).startswith("application/json"):
            data = response.json()
            task_id = data.get("task_id")
        assert task_id, "No task_id found in response or headers"
        status_response = client.get(f"/training/{task_id}/status")
        assert status_response.status_code == HTTP_200_OK
        status_data = status_response.json()
        assert status_data["status"] in {"PENDING", "RUNNING", "DONE", "FAILED"}
        model_dirs = TRAINED_MODELS_DIR.glob("*-finetuned-*")
        for d in model_dirs:
            shutil.rmtree(d, ignore_errors=True)

    @staticmethod
    def test_get_training_status(client: TestClient) -> None:
        """Test the status endpoint for a training task with random ID (should fail)."""
        random_id = str(uuid.uuid4())
        response = client.get(f"/training/{random_id}/status")
        assert response.status_code == HTTP_404_NOT_FOUND
        data = response.json()
        assert "not found" in str(data).lower()

    @staticmethod
    def test_training_with_single_dataset(client: TestClient) -> None:
        """Test training with only one dataset (should succeed)."""
        ensure_minilm_model_available()
        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [DATASET_ID_1],
            "epochs": 1,
            "learning_rate": DEFAULT_LR,
            "per_device_train_batch_size": DEFAULT_BATCH_SIZE,
        }
        response = client.post("/training/train", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED
        task_id = None
        if response.headers.get("Location"):
            match = re.search(
                r"/training/([a-f0-9\-]+)/status",
                response.headers["Location"],
            )
            if match:
                task_id = match.group(1)
        elif response.content and response.headers.get(
            "content-type", ""
        ).startswith("application/json"):
            data = response.json()
            task_id = data.get("task_id")
        assert task_id, "No task_id found in response or headers"
