# ruff: noqa: S101

"""Tests for ZIP model upload functionality."""

from pathlib import Path
from uuid import UUID

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from tests.utils import get_test_zip_file
from vectorize.config.config import settings


@pytest.mark.asyncio
@pytest.mark.upload
class TestZipModelUpload:
    """Tests for uploading models via ZIP files."""

    _base_dir = Path(__file__).parent.parent.parent / "test_data" / "local_models"
    _upload_dir = Path(__file__).parent.parent.parent / settings.model_upload_dir
    _valid_zip = _base_dir / "local_test_model.zip"
    _empty_zip = _base_dir / "empty_model.zip"
    _no_model_zip = _base_dir / "no_model.zip"
    _duplicate_model_zip = _base_dir / "duplicate_model.zip"
    _multiple_models_zip = _base_dir / "multiple_model.zip"
    _filtered_test_zip = _base_dir / "filtered_test_model.zip"

    async def test_valid_zip_upload(self, client: TestClient) -> None:
        """Test uploading a valid ZIP file with model files."""
        files = get_test_zip_file(TestZipModelUpload._valid_zip)

        response = client.post(
            "/uploads/local", params={"model_name": "test_model"}, files=files
        )

        assert response.status_code == status.HTTP_201_CREATED
        assert "Location" in response.headers

        model_id = response.headers["Location"].split("/")[-1]
        try:
            UUID(model_id)
        except ValueError:
            assert model_id == "local_test_model"

        model_dir = Path(self._upload_dir) / "local_test_model"
        assert model_dir.exists(), (
            f"Model directory 'test_model' should exist in {self._upload_dir}"
        )
        assert model_dir.is_dir(), "Model path should be a directory"

        model_files = list(model_dir.glob("*"))
        assert len(model_files) > 0, "Model directory should contain extracted files"

    @staticmethod
    async def test_invalid_file_extension(client: TestClient) -> None:
        """Test uploading a file with an invalid extension."""
        invalid_file_path = TestZipModelUpload._base_dir / "invalid_file.txt"
        if not invalid_file_path.exists():
            with invalid_file_path.open("w") as f:
                f.write("This is not a ZIP file")

        files = get_test_zip_file(invalid_file_path)
        response = client.post(
            "/uploads/local", params={"model_name": "invalid_model"}, files=files
        )

        assert response.status_code != status.HTTP_201_CREATED
        assert response.json()["code"] == "INVALID_FILE"

    @staticmethod
    async def test_empty_zip_upload(client: TestClient) -> None:
        """Test uploading an empty ZIP file."""
        files = get_test_zip_file(TestZipModelUpload._empty_zip)

        response = client.post(
            "/uploads/local",
            params={"model_name": "empty_zip_model", "extract_zip": "false"},
            files=files,
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response.json()["code"] == "INVALID_FILE"

    @staticmethod
    async def test_zip_without_models(client: TestClient) -> None:
        """Test uploading a ZIP file without valid model files."""
        files = get_test_zip_file(TestZipModelUpload._no_model_zip)

        response = client.post(
            "/uploads/local",
            params={"model_name": "no_model_zip", "extract_zip": "true"},
            files=files,
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response.json()["code"] == "INVALID_FILE"

    @staticmethod
    async def test_duplicate_model(client: TestClient) -> None:
        """Test uploading a ZIP file with a copy of the model."""
        files = get_test_zip_file(TestZipModelUpload._duplicate_model_zip)

        response = client.post(
            "/uploads/local",
            params={"model_name": "duplicate_model", "extract_zip": "true"},
            files=files,
        )
        assert response.status_code == status.HTTP_409_CONFLICT
        assert response.json()["code"] == "MODEL_ALREADY_EXISTS"

    @classmethod
    async def test_multiple_model(cls, client: TestClient) -> None:
        """Test uploading a ZIP file with a copy of the model."""
        files = get_test_zip_file(TestZipModelUpload._multiple_models_zip)
        file_count = 2

        response = client.get("/models?size=100")
        assert response.status_code == status.HTTP_200_OK
        models_length = len(response.json()["items"])

        response = client.post(
            "/uploads/local",
            params={"model_name": "multiple_models", "extract_zip": "true"},
            files=files,
        )

        assert response.status_code == status.HTTP_201_CREATED
        response = client.get("/models?size=100")
        assert len(response.json()["items"]) == models_length + file_count

    @staticmethod
    async def test_file_filtering_extraction(client: TestClient) -> None:
        """Test that only model files and JSON files are extracted from ZIP."""
        files = get_test_zip_file(TestZipModelUpload._filtered_test_zip)

        response = client.post(
            "/uploads/local", params={"model_name": "filtered_test_model"}, files=files
        )

        assert response.status_code == status.HTTP_201_CREATED

        model_dir = Path(settings.model_upload_dir) / "filtered_test_model"
        assert model_dir.exists()

        extracted_files = list(model_dir.glob("*"))
        extracted_names = [f.name for f in extracted_files]

        assert "model.bin" in extracted_names
        assert "config.json" in extracted_names

        assert "invalid_file.txt" not in extracted_names
        assert "script.py" not in extracted_names
