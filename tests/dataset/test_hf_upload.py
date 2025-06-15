# ruff: noqa: S101

"""Tests for dataset upload from Hugging Face."""

import asyncio
import time

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from vectorize.common.task_status import TaskStatus
from vectorize.config.config import settings

_TIMEOUT = 60  # seconds
_FILE_SIZE_THRESHOLD = 100


@pytest.mark.asyncio
@pytest.mark.dataset
@pytest.mark.dataset_hf
class TestHuggingFaceUpload:
    """Tests for invalid datasets."""

    @pytest.mark.parametrize(
        "dataset_tag,expected_splits,expected_subsets",
        [
            ("Intel/orca_dpo_pairs", ["train"], ["default"]),
            (
                "argilla/ultrafeedback-binarized-preferences-cleaned",
                ["train"],
                ["default"],
            ),
            ("Dahoas/full-hh-rlhf", ["train", "test"], ["default"]),
            ("argilla/distilabel-intel-orca-dpo-pairs", ["train"], ["default"]),
        ],
    )
    @staticmethod
    async def test_hf_upload_valid(
        client: TestClient,
        dataset_tag: str,
        expected_splits: list[str],
        expected_subsets: list[str],
    ) -> None:
        """Test uploading a valid Hugging Face dataset."""
        response = client.post(
            "/datasets/huggingface", json={"dataset_tag": dataset_tag}
        )
        assert response.status_code == status.HTTP_201_CREATED

        task_id = response.headers["Location"].split("/")[-1]
        task_data = await wait_for_task_completion(client, task_id)

        assert task_data is not None, f"Task {task_id} did not complete"
        assert task_data["id"] == task_id
        assert task_data["task_status"] == "D"
        assert task_data["dataset_tag"] is not None
        assert task_data["error_msg"] is None

        dataset_name_base = dataset_tag.replace("/", "_")

        for subset in expected_subsets:
            for split in expected_splits:
                if subset == "default":
                    file_name = f"{dataset_name_base}_{split}.jsonl"
                else:
                    file_name = f"{dataset_name_base}_{split}_{subset}.jsonl"

                file_path = settings.dataset_upload_dir / file_name
                assert file_path.exists(), (
                    f"File {file_path} was not created for split '{split}', subset '{subset}'"  # noqa: E501
                )

                file_size = file_path.stat().st_size
                assert file_size > 0, f"File {file_path} is empty"
                assert file_size > _FILE_SIZE_THRESHOLD, (
                    f"File {file_path} is too small ({file_size} bytes)"
                )

    @pytest.mark.parametrize(
        "dataset_tag",
        [
            "Mialiu/nerf",
            "Hcompany/WebClick",
            "Jiahao004/DeepTheorem",
            "yandex/alchemist",
        ],
    )
    @staticmethod
    async def test_hf_upload_invalid(client: TestClient, dataset_tag: str) -> None:
        """Test uploading an invalid file format."""
        response = client.post(
            "/datasets/huggingface", json={"dataset_tag": dataset_tag}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "Location" not in response.headers

    @staticmethod
    async def test_hf_upload_not_found(client: TestClient) -> None:
        """Test uploading an invalid file format."""
        response = client.post(
            "/datasets/huggingface",
            json={"dataset_tag": "nonexistent/dataset_tag"},
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "Location" not in response.headers


async def wait_for_task_completion(
    client: TestClient, task_id: str, poll_interval: float = 1.0
) -> dict | None:
    """Poll task status until completion or timeout."""
    start_time = time.time()

    while time.time() - start_time < _TIMEOUT:
        status_response = client.get(f"/datasets/huggingface/status/{task_id}")

        if status_response.status_code == status.HTTP_200_OK:
            data = status_response.json()
            if data["task_status"] in {TaskStatus.DONE, TaskStatus.FAILED}:
                return data

        await asyncio.sleep(poll_interval)

    raise TimeoutError(f"Task {task_id} did not complete within {_TIMEOUT} seconds")
