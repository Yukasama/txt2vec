"""UploadTask model."""
import uuid
from datetime import datetime

from pydantic import BaseModel
from sqlmodel import Column, DateTime, Field, SQLModel, func

from txt2vec.ai_model.model_source import ModelSource
from txt2vec.common.status import TaskStatus


class UploadRequest(BaseModel):
    """Upload process."""
    git_repo: str
    model_tag: str


class UploadResponse(BaseModel):
    """Upload process."""
    upload_id: str
    status: TaskStatus


class StatusResponse(BaseModel):
    """Upload process."""
    upload_id: str
    status: TaskStatus
    error_msg: str | None = None


class UploadTask(SQLModel, table=True):
    """UploadTask model."""

    __tablename__ = "upload_task"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        description="Unique identifier for the upload task.",
    )

    model_tag: str = Field(
        description="Tag of the model file being uploaded.",
        index=True,
        min_length=1,
        max_length=128,
    )

    task_status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        index=True,
        description="Status of the upload task.",
    )

    source: ModelSource = Field(
        description="Source of the model (github or huggingface)."
    )

    end_date: datetime | None = Field(
        default=None,
        description="Optional end time of the synthetic generation.",
    )

    error_msg: str | None = Field(
        default=None,
        description="Optional error message encountered during upload.",
    )

    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), insert_default=func.now()),
        description="Timestamp when the upload task was created.",
    )

    updated_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True), onupdate=func.now(), insert_default=func.now()
        ),
        description="Timestamp when the upload task was last updated.",
    )
