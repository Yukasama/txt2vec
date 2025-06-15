"""UploadDatasetTask model."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlmodel import Column, DateTime, Field, SQLModel, func

from vectorize.common.task_status import TaskStatus


class UploadDatasetTask(SQLModel, table=True):
    """Dataset upload task model for tracking dataset processing operations."""

    __tablename__ = "upload_dataset_task"

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        description="Unique identifier for the dataset upload task.",
    )

    dataset_tag: str = Field(
        description="Tag or identifier of the dataset being uploaded.",
        index=True,
        min_length=1,
        max_length=128,
    )

    task_status: TaskStatus = Field(
        default=TaskStatus.QUEUED,
        index=True,
        description="Current status of the dataset upload task.",
    )

    end_date: datetime | None = Field(
        default=None,
        description="Timestamp when the dataset upload task completed.",
    )

    error_msg: str | None = Field(
        default=None,
        description="Error message if the dataset upload task failed.",
    )

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), insert_default=func.now()),
        description="Timestamp when the dataset upload task was created.",
    )

    updated_at: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True), onupdate=func.now(), insert_default=func.now()
        ),
        description="Timestamp when the dataset upload task was last updated.",
    )
