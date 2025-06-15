"""UploadTask model."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlmodel import Column, DateTime, Field, SQLModel, func

from vectorize.ai_model.model_source import RemoteModelSource
from vectorize.common.task_status import TaskStatus


class UploadTask(SQLModel, table=True):
    """UploadTask model."""

    __tablename__ = "upload_task"

    id: UUID = Field(
        default_factory=uuid4,
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
        default=TaskStatus.RUNNING,
        index=True,
        description="Status of the upload task.",
    )

    source: RemoteModelSource = Field(
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

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), insert_default=func.now()),
        description="Timestamp when the upload task was created.",
    )

    updated_at: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True), onupdate=func.now(), insert_default=func.now()
        ),
        description="Timestamp when the upload task was last updated.",
    )
