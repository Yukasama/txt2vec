"""Repository functions for TrainingTask persistence."""

from datetime import UTC, datetime
from uuid import UUID

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.common.task_status import TaskStatus

from .models import TrainingTask

__all__ = [
    "get_train_task_by_id",
    "save_training_task",
    "update_training_task_progress",
    "update_training_task_status",
]


async def save_training_task(db: AsyncSession, task: TrainingTask) -> None:
    """Persist a new TrainingTask to the database.

    Args:
        db: The database session.
        task: The training task to save.
    """
    db.add(task)
    await db.commit()
    await db.refresh(task)


async def update_training_task_status(
    db: AsyncSession, task_id: UUID, status: TaskStatus, error_msg: str | None = None
) -> None:
    """Update the status and error message of a TrainingTask.

    Args:
        db: The database session.
        task_id (UUID): The ID of the training task.
        status (TaskStatus): The new status.
        error_msg (str | None): Optional error message.
    """
    result = await db.exec(select(TrainingTask).where(TrainingTask.id == task_id))
    task = result.first()
    if task:
        task.task_status = status
        if status in {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELED}:
            task.end_date = datetime.now(UTC)
        if error_msg:
            task.error_msg = error_msg
        await db.commit()
        await db.refresh(task)


async def update_training_task_progress(
    db: AsyncSession, task_id: UUID, progress: float
) -> None:
    """Update the progress of a TrainingTask.

    Args:
        db (AsyncSession): The database session.
        task_id (UUID): The ID of the training task.
        progress (float): The new progress value (0.0 to 1.0).
    """
    result = await db.exec(select(TrainingTask).where(TrainingTask.id == task_id))
    task = result.first()
    if task:
        task.progress = progress
        await db.commit()
        await db.refresh(task)


async def get_train_task_by_id(db: AsyncSession, task_id: UUID) -> TrainingTask | None:
    """Fetch a TrainingTask by its ID.

    Args:
        db (AsyncSession): The database session.
        task_id (UUID): The ID of the training task.

    Returns:
        TrainingTask | None: The training task if found, else None.
    """
    result = await db.exec(select(TrainingTask).where(TrainingTask.id == task_id))
    return result.first()
