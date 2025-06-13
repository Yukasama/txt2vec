"""Tasks module for handling background processes.

This module contains functions for managing background tasks related to
model uploads and processing, such as handling Hugging Face models.
"""

from uuid import UUID

from loguru import logger
from sqlalchemy.exc import IntegrityError
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.ai_model import AIModel
from vectorize.ai_model.models import ModelSource
from vectorize.ai_model.repository import save_ai_model_db
from vectorize.common.task_status import TaskStatus

from .exceptions import ModelAlreadyExistsError
from .github_service import load_github_model_and_cache_only_svc
from .huggingface_service import load_huggingface_model_and_cache_only_svc
from .repository import update_upload_task_status_db

__all__ = [
    "process_github_model_bg",
    "process_huggingface_model_bg",
]


async def process_huggingface_model_bg(
    db: AsyncSession, model_tag: str, revision: str, task_id: UUID
) -> None:
    """Processes a Hugging Face model upload in the background.

    This function handles the background processing of a Hugging Face model
    upload task. It loads the model, saves it to the database, and updates
    the task status.

    Args:
        db (AsyncSession): The database session for database operations.
        model_tag (str): The tag of the Hugging Face model repository.
        revision (str): The specific revision or version of the model to download.
        task_id (UUID): The unique identifier of the upload task.

    Raises:
        ModelAlreadyExistsError: If the model already exists in the database.
        IntegrityError: If a database integrity error occurs.
        Exception: If an error occurs during model processing or database operations.
    """
    key = f"{model_tag}@{revision}"

    try:
        logger.info("[BG] Starting model upload for task", taskId=task_id)
        await load_huggingface_model_and_cache_only_svc(model_tag, revision)

        ai_model = AIModel(
            model_tag=key,
            name=model_tag,
            source=ModelSource.HUGGINGFACE,
        )
        await save_ai_model_db(db, ai_model)
        await update_upload_task_status_db(db, task_id, TaskStatus.DONE)

        logger.info("[BG] Task completed successfully", taskId=task_id)

    except ModelAlreadyExistsError as e:
        logger.error(f"[BG] Model already exists for task {task_id}: {e}")
        await db.rollback()
        await update_upload_task_status_db(
            db, task_id, TaskStatus.FAILED, error_msg=str(e)
        )
    except IntegrityError as e:
        logger.error(f"[BG] IntegrityError in task {task_id}: {e}")
        await db.rollback()
        await update_upload_task_status_db(
            db, task_id, TaskStatus.FAILED, error_msg=str(e)
        )
    except Exception as e:
        logger.error(f"[BG] Error in task {task_id}: {e}")
        await db.rollback()
        await update_upload_task_status_db(
            db, task_id, TaskStatus.FAILED, error_msg=str(e)
        )


async def process_github_model_bg(  # noqa: D417
    db: AsyncSession, owner: str, repo: str, branch: str, task_id: UUID
) -> None:
    """Processes a GitHub model upload in the background.

    This function handles the background processing of a GitHub model
    upload task. It loads the model, saves it to the database, and updates
    the task status.

    Args:
        db (AsyncSession): The database session for database operations.
        owner (str):
        repo (str):
        branch (str):
        task_id (UUID): The unique identifier of the upload task.

    Raises:
        Exception: If an error occurs during model processing or database
        operations.
    """
    key = f"{owner}/{repo}@{branch}"

    try:
        logger.info("[BG] Starting model upload for task", taskId=task_id)
        load_github_model_and_cache_only_svc(owner, repo, branch)

        ai_model = AIModel(
            model_tag=key,
            name=repo,
            source=ModelSource.GITHUB,
        )
        await save_ai_model_db(db, ai_model)
        await update_upload_task_status_db(db, task_id, TaskStatus.DONE)

        logger.info("[BG] Task completed successfully", taskId=task_id)

    except ModelAlreadyExistsError as e:
        logger.error(f"[BG] Model already exists for task {task_id}: {e}")
        await db.rollback()
        await update_upload_task_status_db(
            db, task_id, TaskStatus.FAILED, error_msg=str(e)
        )
    except IntegrityError as e:
        logger.error(f"[BG] IntegrityError in task {task_id}: {e}")
        await db.rollback()
        await update_upload_task_status_db(
            db, task_id, TaskStatus.FAILED, error_msg=str(e)
        )
    # pylint: disable=broad-except
    except Exception as e:
        logger.error(f"[BG] Error in task {task_id}: {e}")
        await db.rollback()
        await update_upload_task_status_db(
            db, task_id, TaskStatus.FAILED, error_msg=str(e)
        )
