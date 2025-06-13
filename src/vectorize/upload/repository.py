"""Repository module for managing upload tasks in the database.

This module provides functions to save and update upload tasks in the database
using SQLModel and asynchronous database sessions.
"""

from datetime import UTC, datetime
from uuid import UUID

from loguru import logger
from sqlalchemy.exc import NoResultFound
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.common.task_status import TaskStatus

from .models import UploadTask

__all__ = [
    "get_upload_task_by_id_db",
    "save_upload_task_db",
    "update_upload_task_status_db"
]


async def save_upload_task_db(db: AsyncSession, task: UploadTask) -> None:
    """Saves a new upload task to the database.

    Args:
        db (AsyncSession): The asynchronous database session.
        task (UploadTask): The upload task to be saved.
    """
    db.add(task)
    await db.commit()
    await db.refresh(task)


async def update_upload_task_status_db(
    db: AsyncSession,
    task_id: UUID,
    status: TaskStatus,
    error_msg: str | None = None,
) -> None:
    """Updates the status of an existing upload task in the database.

    Args:
        db (AsyncSession): The asynchronous database session.
        task_id (UUID): The unique identifier of the upload task.
        status (TaskStatus): The new status to set for the task.
        error_msg (str | None): An optional error message if the task failed.

    Hinweis: Falls der Task nicht gefunden wird, wird ein Fehler geloggt,
    aber keine Exception geworfen.

    Raises:
        NoResultFound: If the task with the given ID is not found (error is only
            logged, not raised).
    """
    result = await db.exec(select(UploadTask).where(UploadTask.id == task_id))
    try:
        task = result.one()
        task.task_status = status
        task.end_date = datetime.now(tz=UTC)
        task.error_msg = error_msg
        await db.commit()
    except NoResultFound:
        logger.exception(
            "[update_upload_task_status_db] Task mit ID nicht gefunden!",
            task_id=task_id
        )


async def get_upload_task_by_id_db(
    db: AsyncSession,
    task_id: UUID
) -> UploadTask | None:
    """Retrieves a single upload task by its ID.

    Args:
        db (AsyncSession): The asynchronous database session.
        task_id (UUID): The unique identifier of the upload task.

    Returns:
        UploadTask | None: The found task, or None if not found.
    """
    result = await db.exec(select(UploadTask).where(UploadTask.id == task_id))
    return result.one_or_none()
