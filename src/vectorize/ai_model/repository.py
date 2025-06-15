"""AIModel repository."""

import os
from collections.abc import Sequence
from uuid import UUID

from huggingface_hub import snapshot_download
from loguru import logger
from sqlmodel import func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.common.exceptions import VersionMismatchError

from .exceptions import ModelNotFoundError, NoModelFoundError
from .models import AIModel, AIModelUpdate, ModelSource

__all__ = ["get_ai_model_db", "save_ai_model_db", "update_ai_model_db"]


async def get_models_paged_db(
    db: AsyncSession,
    page: int = 1,
    size: int = 5,
) -> tuple[Sequence[AIModel], int]:
    """Fetches a page of AIModel entries from the database.

    Args:
        db (AsyncSession): The database session.
        page (int, optional): Page number, starts at 1. Defaults to 1.
        size (int, optional): Number of items per page. Defaults to 5.

    Returns:
        tuple[list[AIModel], int]: A tuple containing the list of AIModel
        objects for the requested page,
        and the total number of models in the database.

    Raises:
        NoModelFoundError: If there are no models in the database.
    """
    total_stmt = select(func.count()).select_from(AIModel)
    total = await db.scalar(total_stmt)

    if not total:
        raise NoModelFoundError()

    offset = (page - 1) * size
    stmt = select(AIModel).offset(offset).limit(size)
    result = await db.exec(stmt)
    items = result.all()

    logger.debug("AIModels retrieved", items_fetched=len(items), total_items=total)
    return items, total


async def get_ai_model_db(db: AsyncSession, model_tag: str) -> AIModel:
    """Retrieve an AI model by its ID.

    Args:
        db: Database session instance.
        model_tag: The Tag of the model to retrieve.

    Returns:
        AIModel: The AI model object corresponding to the given Model Tag.

    Raises:
        ModelNotFoundError: If the model is not found.
    """
    statement = select(AIModel).where(AIModel.model_tag == model_tag)
    result = await db.exec(statement)
    model = result.first()

    if model is None:
        raise ModelNotFoundError(model_tag)

    logger.debug("AI Model loaded from DB", ai_model=model)
    return model


async def save_ai_model_db(db: AsyncSession, model: AIModel) -> UUID:
    """Persist model to database with a unique tag.

    Args:
        db: Database session instance.
        model: The AI model object to save.

    Returns:
        UUID: The ID of the saved AI model.
    """
    db.add(model)
    await db.commit()
    await db.refresh(model)

    logger.debug("AI Model saved to DB", ai_model=model)
    return model.id


async def update_ai_model_db(
    db: AsyncSession, model_id: UUID, update_data: AIModelUpdate, expected_version: int
) -> AIModel:
    """Update an AIModel using optimistic locking.

    Args:
        db: The database session instance.
        model_id: The unique identifier of the AI model to update.
        update_data: The Pydantic model containing the fields to update.
        expected_version: The version expected by the client (for optimistic locking).

    Returns:
        AIModel: The updated AI model object.

    Raises:
        ModelNotFoundError: If the AI model is not found.
        VersionMismatchError: If the version does not match (lost update).
    """
    result = await db.exec(select(AIModel).where(AIModel.id == model_id))
    model = result.first()
    if model is None:
        raise ModelNotFoundError(model_id)

    if model.version != expected_version:
        raise VersionMismatchError(model_id, model.version)

    for field, value in update_data.model_dump(exclude_unset=True).items():
        setattr(model, field, value)

    model.version += 1

    await db.commit()
    await db.refresh(model)

    logger.debug("AIModel updated", ai_model=model)
    return model


async def delete_model_db(db: AsyncSession, model_id: UUID) -> None:
    """Delete an AI model from the database by ID.

    Args:
        db: Database session instance.
        model_id: The UUID of the AI model to delete.

    Raises:
        ModelNotFoundError: If the model with the given ID does not exist.
    """
    statement = select(AIModel).where(AIModel.id == model_id)
    result = await db.exec(statement)
    model = result.first()

    if model is None:
        raise ModelNotFoundError(str(model_id))

    await db.delete(model)
    await db.commit()
    logger.debug("Model deleted", model=model)


async def get_ai_model_by_id(db: AsyncSession, model_id: UUID) -> AIModel:
    """Retrieve an AI model by its UUID."""
    statement = select(AIModel).where(AIModel.id == model_id)
    result = await db.exec(statement)
    model = result.first()
    if model is None:
        raise ModelNotFoundError(str(model_id))
    logger.debug("AI Model loaded from DB by ID", ai_model=model)
    return model


async def import_hf_model_and_register(repo_id: str, db: AsyncSession) -> UUID:
    """Download a Hugging Face model, store the snapshot folder as model_tag."""
    snapshot_path = snapshot_download(
        repo_id,
        local_dir="data/models",
        local_dir_use_symlinks=False,
    )
    model_tag = os.path.relpath(snapshot_path, "data/models")
    model = AIModel(
        name=repo_id,
        model_tag=model_tag,
        source=ModelSource.HUGGINGFACE,
    )
    await save_ai_model_db(db, model)
    return model.id
