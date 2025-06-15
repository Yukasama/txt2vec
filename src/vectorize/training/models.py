"""Training Task model."""

from datetime import datetime
from typing import TYPE_CHECKING, Optional
from uuid import UUID, uuid4

from sqlmodel import Column, DateTime, Field, Relationship, SQLModel, func

from vectorize.common.task_status import TaskStatus

if TYPE_CHECKING:
    from vectorize.ai_model.models import AIModel

__all__ = ["TrainingTask"]


class TrainingTask(SQLModel, table=True):
    """Model training task."""

    __tablename__ = "training_task"

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        description="Unique identifier for the training task.",
    )

    task_status: TaskStatus = Field(
        default=TaskStatus.QUEUED,
        index=True,
        description="Current status of the model training.",
    )

    end_date: datetime | None = Field(
        default=None, description="Optional end time of the training task."
    )

    error_msg: str | None = Field(
        default=None,
        description="Optional error message encountered during training.",
    )

    trained_model_id: UUID | None = Field(
        default=None,
        foreign_key="ai_model.id",
        description="ID of the trained AI model.",
    )

    trained_model: Optional["AIModel"] = Relationship(back_populates="training_tasks")

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), insert_default=func.now()),
        description="Timestamp when the training task was created.",
    )

    updated_at: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True), onupdate=func.now(), insert_default=func.now()
        ),
        description="Timestamp when the training task was last updated.",
    )

    progress: float = Field(
        default=0.0,
        description="Progress of the training task as a float between 0.0 and 1.0.",
    )
