"""Service layer for model evaluation."""

import json
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from uuid import UUID

from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.ai_model.exceptions import ModelNotFoundError
from vectorize.ai_model.repository import get_ai_model_db
from vectorize.common.task_status import TaskStatus
from vectorize.dataset.repository import get_dataset_db
from vectorize.training.exceptions import (
    InvalidDatasetIdError,
    TrainingDatasetNotFoundError,
)
from vectorize.training.utils.uuid_validator import is_valid_uuid

from .evaluation import TrainingEvaluator
from .repository import update_evaluation_task_results, update_evaluation_task_status
from .schemas import EvaluationRequest, EvaluationResponse
from .utils import resolve_model_path

__all__ = ["evaluate_model_background_task", "evaluate_model_task"]


async def evaluate_model_task(
    db: AsyncSession,
    evaluation_request: EvaluationRequest,
) -> EvaluationResponse:
    """Service entry point for model evaluation.

    Args:
        db (AsyncSession): Database session.
        evaluation_request (EvaluationRequest): Evaluation configuration.

    Returns:
        EvaluationResponse: Evaluation results.

    Raises:
        InvalidDatasetIdError: If dataset ID is invalid.
        TrainingDatasetNotFoundError: If dataset file not found.
        ModelNotFoundError: If model tag not found.
    """
    model = await get_ai_model_db(db, evaluation_request.model_tag)
    if not model:
        raise ModelNotFoundError(evaluation_request.model_tag)

    model_path = resolve_model_path(model.model_tag)

    if not is_valid_uuid(evaluation_request.dataset_id):
        raise InvalidDatasetIdError(evaluation_request.dataset_id)

    dataset_uuid = UUID(evaluation_request.dataset_id)
    dataset = await get_dataset_db(db, dataset_uuid)
    if not dataset:
        raise TrainingDatasetNotFoundError(
            f"Dataset not found: {evaluation_request.dataset_id}"
        )

    dataset_path = Path("data/datasets") / dataset.file_name
    if not dataset_path.exists():
        raise TrainingDatasetNotFoundError(f"Dataset file not found: {dataset_path}")

    evaluator = TrainingEvaluator(model_path)

    if evaluation_request.baseline_model_tag:
        baseline_model = await get_ai_model_db(
            db, evaluation_request.baseline_model_tag
        )
        if not baseline_model:
            raise ModelNotFoundError(evaluation_request.baseline_model_tag)

        baseline_model_path = resolve_model_path(baseline_model.model_tag)

        comparison_results = evaluator.compare_models(
            baseline_model_path=baseline_model_path,
            dataset_path=dataset_path,
            max_samples=evaluation_request.max_samples,
        )
        trained_metrics = comparison_results["trained"]
        baseline_metrics = comparison_results["baseline"]

        improvement = trained_metrics.get_improvement_over_baseline(baseline_metrics)
        success_status = (
            "successful" if trained_metrics.is_training_successful() else "unsuccessful"
        )
        summary = (
            f"Training {success_status} "
            f"({trained_metrics.get_quality_grade()} quality). "
            f"Similarity ratio improved by {improvement['ratio_improvement']:.3f} "
            f"({baseline_metrics.similarity_ratio:.3f} → "
            f"{trained_metrics.similarity_ratio:.3f})"
        )

        return EvaluationResponse(
            model_tag=evaluation_request.model_tag,
            dataset_used=str(dataset_path),
            metrics=trained_metrics.to_dict(),
            baseline_metrics=baseline_metrics.to_baseline_dict(),
            evaluation_summary=summary,
            training_successful=trained_metrics.is_training_successful(),
        )
    metrics = evaluator.evaluate_dataset(dataset_path, evaluation_request.max_samples)

    success_status = (
        "successful" if metrics.is_training_successful() else "unsuccessful"
    )
    summary = (
        f"Training {success_status} ({metrics.get_quality_grade()} quality). "
        f"Positive similarity: {metrics.avg_positive_similarity:.3f}, "
        f"Negative similarity: {metrics.avg_negative_similarity:.3f}, "
        f"Ratio: {metrics.similarity_ratio:.3f}"
    )

    return EvaluationResponse(
        model_tag=evaluation_request.model_tag,
        dataset_used=str(dataset_path),
        metrics=metrics.to_dict(),
        evaluation_summary=summary,
        training_successful=metrics.is_training_successful(),
    )


async def evaluate_model_background_task(
    get_session_func: Callable[[], AsyncGenerator[AsyncSession]],
    evaluation_request: EvaluationRequest,
    task_id: UUID,
) -> None:
    """Background task for model evaluation.

    Args:
        get_session_func: Function that returns a database session
        evaluation_request: Evaluation configuration
        task_id: ID of the evaluation task
    """
    async for db in get_session_func():
        try:
            logger.debug(
                "Starting background evaluation",
                task_id=str(task_id),
                model_tag=evaluation_request.model_tag,
                dataset_id=evaluation_request.dataset_id,
                baseline_model_tag=evaluation_request.baseline_model_tag,
                max_samples=evaluation_request.max_samples,
            )

            await update_evaluation_task_status(
                db, task_id, TaskStatus.RUNNING, progress=0.1
            )

            model = await get_ai_model_db(db, evaluation_request.model_tag)
            if not model:
                raise ModelNotFoundError(evaluation_request.model_tag)

            model_path = resolve_model_path(model.model_tag)

            if not is_valid_uuid(evaluation_request.dataset_id):
                raise InvalidDatasetIdError(evaluation_request.dataset_id)

            dataset_uuid = UUID(evaluation_request.dataset_id)
            dataset = await get_dataset_db(db, dataset_uuid)
            if not dataset:
                raise TrainingDatasetNotFoundError(
                    f"Dataset not found: {evaluation_request.dataset_id}"
                )

            dataset_path = Path("data/datasets") / dataset.file_name
            if not dataset_path.exists():
                raise TrainingDatasetNotFoundError(
                    f"Dataset file not found: {dataset_path}"
                )

            await update_evaluation_task_status(
                db, task_id, TaskStatus.RUNNING, progress=0.3
            )

            evaluator = TrainingEvaluator(model_path)

            if evaluation_request.baseline_model_tag:
                baseline_model = await get_ai_model_db(
                    db, evaluation_request.baseline_model_tag
                )
                if not baseline_model:
                    raise ModelNotFoundError(evaluation_request.baseline_model_tag)

                baseline_model_path = resolve_model_path(baseline_model.model_tag)

                await update_evaluation_task_status(
                    db, task_id, TaskStatus.RUNNING, progress=0.5
                )

                comparison_results = evaluator.compare_models(
                    baseline_model_path=baseline_model_path,
                    dataset_path=dataset_path,
                    max_samples=evaluation_request.max_samples,
                )
                trained_metrics = comparison_results["trained"]
                baseline_metrics = comparison_results["baseline"]

                improvement = trained_metrics.get_improvement_over_baseline(
                    baseline_metrics
                )
                success_status = (
                    "successful"
                    if trained_metrics.is_training_successful()
                    else "unsuccessful"
                )
                summary = (
                    f"Training {success_status} "
                    f"({trained_metrics.get_quality_grade()} quality). "
                    f"Similarity ratio improved by "
                    f"{improvement['ratio_improvement']:.3f} "
                    f"({baseline_metrics.similarity_ratio:.3f} → "
                    f"{trained_metrics.similarity_ratio:.3f})"
                )

                await update_evaluation_task_status(
                    db, task_id, TaskStatus.RUNNING, progress=0.9
                )

                await update_evaluation_task_results(
                    db,
                    task_id,
                    evaluation_metrics=json.dumps(trained_metrics.to_dict()),
                    baseline_metrics=json.dumps(baseline_metrics.to_baseline_dict()),
                    evaluation_summary=summary,
                )

            else:
                await update_evaluation_task_status(
                    db, task_id, TaskStatus.RUNNING, progress=0.5
                )

                metrics = evaluator.evaluate_dataset(
                    dataset_path, evaluation_request.max_samples
                )

                success_status = (
                    "successful"
                    if metrics.is_training_successful()
                    else "unsuccessful"
                )
                summary = (
                    f"Training {success_status} "
                    f"({metrics.get_quality_grade()} quality). "
                    f"Positive similarity: {metrics.avg_positive_similarity:.3f}, "
                    f"Negative similarity: {metrics.avg_negative_similarity:.3f}, "
                    f"Ratio: {metrics.similarity_ratio:.3f}"
                )

                await update_evaluation_task_status(
                    db, task_id, TaskStatus.RUNNING, progress=0.9
                )

                await update_evaluation_task_results(
                    db,
                    task_id,
                    evaluation_metrics=json.dumps(metrics.to_dict()),
                    evaluation_summary=summary,
                )

            await update_evaluation_task_status(
                db, task_id, TaskStatus.DONE, progress=1.0
            )

            logger.debug(
                "Background evaluation completed successfully",
                task_id=str(task_id),
                model_tag=evaluation_request.model_tag,
                dataset_path=str(dataset_path),
                has_baseline=evaluation_request.baseline_model_tag is not None,
            )
            break

        except Exception as exc:
            logger.error(
                "Background evaluation failed",
                task_id=str(task_id),
                model_tag=evaluation_request.model_tag,
                dataset_id=evaluation_request.dataset_id,
                error=str(exc),
                exc_info=exc,
            )
            await update_evaluation_task_status(
                db, task_id, TaskStatus.FAILED, error_msg=str(exc)
            )
            break
