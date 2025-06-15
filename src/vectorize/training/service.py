"""Service layer for training orchestration. Delegates to orchestrator and utils."""

from uuid import UUID

from sqlmodel.ext.asyncio.session import AsyncSession

from .schemas import TrainRequest
from .training_orchestrator import run_training


async def train_model_task(  # noqa: PLR0913, PLR0917
    db: AsyncSession,
    model_path: str,
    train_request: TrainRequest,
    task_id: UUID,
    dataset_paths: list[str],
    output_dir: str,
) -> None:
    """Service entry point for SBERT training. Delegates to orchestrator.

    Args:
        db (AsyncSession): Database session.
        model_path (str): Path to the base model.
        train_request (TrainRequest): Training configuration.
        task_id (UUID): Training task ID.
        dataset_paths (list[str]): List of dataset file paths.
        output_dir (str): Output directory for the trained model.
    """
    await run_training(
        db=db,
        model_path=model_path,
        train_request=train_request,
        task_id=task_id,
        dataset_paths=dataset_paths,
        output_dir=output_dir,
    )
