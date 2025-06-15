"""Training router for SBERT triplet training (Hugging Face TRL)."""

import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, Response, status
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.ai_model.exceptions import ModelNotFoundError
from vectorize.ai_model.repository import get_ai_model_db
from vectorize.common.task_status import TaskStatus
from vectorize.config.db import get_session
from vectorize.dataset.repository import get_dataset_db

from .exceptions import (
    InvalidDatasetIdError,
    InvalidModelIdError,
    TrainingDatasetNotFoundError,
    TrainingModelWeightsNotFoundError,
    TrainingTaskNotFoundError,
)
from .models import TrainingTask
from .repository import (
    get_train_task_by_id,
    save_training_task,
    update_training_task_status,
)
from .schemas import TrainRequest, TrainingStatusResponse
from .service import train_model_task
from .utils.uuid_validator import is_valid_uuid

__all__ = ["router"]

router = APIRouter(tags=["Training"])


@router.post("/train")
async def train_model(  # noqa: PLR0914, PLR0915
    train_request: TrainRequest,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Starts SBERT triplet training as a background task.

    Expects JSONL with question, positive, negative.
    If no val_dataset_id is provided,
    10% of the first training dataset will be used for validation.
    """
    if not hasattr(train_request, "model_tag"):
        raise InvalidModelIdError("TrainRequest muss ein model_tag enthalten!")
    model = await get_ai_model_db(db, train_request.model_tag)
    if not model:
        raise ModelNotFoundError(train_request.model_tag)
    model_path = str(Path("data/models") / model.model_tag)

    def has_model_weights(path: str) -> bool:
        for _root, _dirs, files in os.walk(path):
            for file in files:
                if file.endswith((".safetensors", ".bin")):
                    return True
        return False

    if not has_model_weights(model_path):
        raise TrainingModelWeightsNotFoundError(
            f"Model weights not found in {model_path} (searched recursively)"
        )
    dataset_paths = []
    required_columns = {"Question", "Positive", "Negative"}
    for _idx, train_ds_id in enumerate(train_request.train_dataset_ids):
        if not is_valid_uuid(train_ds_id):
            raise InvalidDatasetIdError(train_ds_id)
        train_ds_uuid = uuid.UUID(train_ds_id)
        train_ds = await get_dataset_db(db, train_ds_uuid)
        train_dataset_path = Path("data/datasets") / train_ds.file_name
        try:
            df = pd.read_json(train_dataset_path, lines=True)
        except Exception as exc:
            raise TrainingDatasetNotFoundError(
                f"Dataset {train_dataset_path} is not a valid JSONL file: {exc}"
            ) from exc
        if not required_columns.issubset(df.columns):
            missing_cols = required_columns - set(df.columns)
            raise TrainingDatasetNotFoundError(
                f"Dataset {train_dataset_path} is missing required columns: "
                f"{missing_cols}"
            )
        dataset_paths.append(str(train_dataset_path))
    if train_request.val_dataset_id:
        val_ds_id = train_request.val_dataset_id
        if not is_valid_uuid(val_ds_id):
            raise InvalidDatasetIdError(val_ds_id)
        val_ds_uuid = uuid.UUID(val_ds_id)
        val_ds = await get_dataset_db(db, val_ds_uuid)
        val_dataset_path = Path("data/datasets") / val_ds.file_name
        try:
            val_df = pd.read_json(val_dataset_path, lines=True)
        except Exception as exc:
            raise TrainingDatasetNotFoundError(
                f"Validation dataset {val_dataset_path} is not a valid JSONL file: "
                f"{exc}"
            ) from exc
        if not required_columns.issubset(val_df.columns):
            missing_cols = required_columns - set(val_df.columns)
            raise TrainingDatasetNotFoundError(
                f"Validation dataset {val_dataset_path} is missing required columns: "
                f"{missing_cols}"
            )
        dataset_paths.append(str(val_dataset_path))
    missing = [str(p) for p in dataset_paths if not Path(p).is_file()]
    if missing:
        raise TrainingDatasetNotFoundError(f"Missing datasets: {', '.join(missing)}")
    tag_time = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    task = TrainingTask(id=uuid4())
    output_dir = (
        f"data/models/trained_models/{model.model_tag}-finetuned-"
        f"{tag_time}-{str(task.id)[:8]}"
    )
    logger.debug(
        "SBERT-Triplet-Training requested."
        "model_tag={}, dataset_count={}, model_path={}",
        train_request.model_tag,
        len(dataset_paths),
        model_path,
    )
    await save_training_task(db, task)
    background_tasks.add_task(
        train_model_task,
        db,
        model_path,
        train_request,
        task.id,
        dataset_paths,
        output_dir,
    )
    logger.debug(
        "SBERT-Triplet-Training started in background.",
        task_id=str(task.id),
        model_tag=train_request.model_tag,
        dataset_count=len(dataset_paths),
        model_path=model_path,
    )
    location = f"/training/{task.id}/status"
    return Response(
        status_code=status.HTTP_202_ACCEPTED,
        headers={"Location": location},
    )


@router.get("/{task_id}/status")
async def get_training_status(
    task_id: UUID,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> TrainingStatusResponse:
    """Returns the status and metadata of a training job."""
    task = await get_train_task_by_id(db, task_id)
    if not task:
        raise TrainingTaskNotFoundError(str(task_id))
    return TrainingStatusResponse.from_task(task)


@router.post("/{task_id}/cancel")
async def cancel_training(
    task_id: UUID,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Cancel a running training task."""
    task = await get_train_task_by_id(db, task_id)
    if not task:
        raise TrainingTaskNotFoundError(str(task_id))
    if task.task_status not in {TaskStatus.QUEUED, TaskStatus.RUNNING}:
        return Response(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=f"Cannot cancel task with status: {task.task_status.name}",
        )
    await update_training_task_status(
        db, task_id, TaskStatus.CANCELED, error_msg="Training canceled by user"
    )
    logger.debug(
        "Training task marked for cancellation.",
        task_id=task_id,
    )
    return Response(status_code=status.HTTP_200_OK)
