"""Evaluation exceptions."""

from fastapi import status

from vectorize.common.app_error import AppError
from vectorize.config.errors import ErrorCode

__all__ = ["EvaluationTaskNotFoundError"]


class EvaluationTaskNotFoundError(AppError):
    """Raised when evaluation task is not found."""

    def __init__(self, task_id: str) -> None:
        """Initialize exception.

        Args:
            task_id: The task ID that was not found
        """
        super().__init__(
            message=f"Evaluation task not found: {task_id}",
            error_code=ErrorCode.NOT_FOUND,
            status_code=status.HTTP_404_NOT_FOUND,
        )
