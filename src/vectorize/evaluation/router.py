"""Router for model evaluation endpoints."""

from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, Response, status
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.common.task_status import TaskStatus
from vectorize.config.db import get_session

from .exceptions import EvaluationTaskNotFoundError
from .models import EvaluationTask
from .repository import get_evaluation_task_by_id, save_evaluation_task
from .schemas import EvaluationRequest, EvaluationStatusResponse
from .service import evaluate_model_background_task

__all__ = ["router"]

router = APIRouter(tags=["Evaluation"])


@router.post("/evaluate")
async def evaluate_model(
    evaluation_request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Start model evaluation as a background task.

    Computes cosine similarity metrics between question-positive-negative triplets
    to assess training quality. Optionally compares against a baseline model.

    Main metrics computed:
    - Average cosine similarity between question and positive examples
    - Average cosine similarity between question and negative examples
    - Ratio of positive to negative similarities (should be > 1)
    - Spearman correlation for similarity ranking

    A training is considered successful if:
    - Positive similarities > negative similarities
    - Similarity ratio > 1.2

    Returns:
        202 Accepted with Location header pointing to task status
    """
    task = EvaluationTask(id=uuid4(), task_status=TaskStatus.PENDING)
    await save_evaluation_task(db, task)

    background_tasks.add_task(
        evaluate_model_background_task,
        get_session,
        evaluation_request,
        task.id,
    )

    logger.debug(
        "Evaluation started in background",
        task_id=str(task.id),
        model_tag=evaluation_request.model_tag,
        dataset_id=evaluation_request.dataset_id,
    )

    location = f"/evaluation/{task.id}/status"
    return Response(
        status_code=status.HTTP_202_ACCEPTED,
        headers={"Location": location},
    )


@router.get("/{task_id}/status")
async def get_evaluation_status(
    task_id: UUID,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> EvaluationStatusResponse:
    """Get the status and results of an evaluation task."""
    task = await get_evaluation_task_by_id(db, task_id)
    if not task:
        raise EvaluationTaskNotFoundError(str(task_id))
    return EvaluationStatusResponse.from_task(task)
