"""Training exceptions."""

from fastapi import status

from vectorize.common.app_error import AppError
from vectorize.config.errors import ErrorCode

__all__ = [
    "DatasetValidationError",
    "InvalidDatasetIdError",
    "InvalidModelIdError",
    "TrainingDatasetNotFoundError",
    "TrainingModelWeightsNotFoundError",
    "TrainingTaskNotFoundError",
]


class TrainingDatasetNotFoundError(AppError):
    """Exception raised when the training dataset file is not found."""

    error_code = ErrorCode.NOT_FOUND
    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, dataset_path: str) -> None:
        """Initialize with the dataset file path."""
        super().__init__(f"Training dataset file not found: {dataset_path}")


class TrainingModelWeightsNotFoundError(AppError):
    """Exception raised when model weights (.bin/.safetensors) are invalid."""

    error_code = ErrorCode.NOT_FOUND
    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, model_path: str) -> None:
        """Initialize the exception for missing or invalid model weights."""
        super().__init__(
            f"Model weights (.bin/.safetensors) not found or invalid in: {model_path}"
        )


class TrainingTaskNotFoundError(AppError):
    """Exception raised when a TrainingTask with the given ID is not found."""

    error_code = ErrorCode.NOT_FOUND
    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, task_id: str) -> None:
        """Initialize with the missing training task ID."""
        super().__init__(f"Training task not found: {task_id}")


class InvalidModelIdError(AppError):
    """Exception raised when a model_tag is missing or invalid."""

    error_code = ErrorCode.NOT_FOUND
    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, model_tag: str) -> None:
        """Initialize with the invalid model tag."""
        super().__init__(f"Model Tag is not valid or missing: {model_tag}")


class InvalidDatasetIdError(AppError):
    """Exception raised when a dataset_id is not a valid UUID."""

    error_code = ErrorCode.NOT_FOUND
    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, dataset_id: str) -> None:
        """Initialize with the invalid dataset ID."""
        super().__init__(f"Dataset ID is not a valid UUID: {dataset_id}")


class DatasetValidationError(AppError):
    """Exception raised when the training dataset is invalid."""

    error_code = ErrorCode.INVALID_FILE
    status_code = status.HTTP_400_BAD_REQUEST

    def __init__(self, msg: str) -> None:
        """Initialize with a validation error message."""
        super().__init__(f"Dataset validation failed: {msg}")
