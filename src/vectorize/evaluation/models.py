"""Evaluation Task model."""

from datetime import datetime
from typing import TYPE_CHECKING, Optional
from uuid import UUID, uuid4

from sqlmodel import Column, DateTime, Field, Relationship, SQLModel, func

from vectorize.common.task_status import TaskStatus

if TYPE_CHECKING:
    from vectorize.ai_model.models import AIModel

__all__ = ["EvaluationTask"]


class EvaluationTask(SQLModel, table=True):
    """Model evaluation task."""

    __tablename__ = "evaluation_task"

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        description="Unique identifier for the evaluation task.",
    )

    task_status: TaskStatus = Field(
        default=TaskStatus.QUEUED,
        index=True,
        description="Current status of the model evaluation.",
    )

    end_date: datetime | None = Field(
        default=None, description="Optional end time of the evaluation task."
    )

    error_msg: str | None = Field(
        default=None,
        description="Optional error message encountered during evaluation.",
    )

    model_id: UUID | None = Field(
        default=None,
        foreign_key="ai_model.id",
        description="ID of the evaluated AI model.",
    )

    model: Optional["AIModel"] = Relationship(back_populates="evaluation_tasks")

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), insert_default=func.now()),
        description="Timestamp when the evaluation task was created.",
    )

    updated_at: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True), onupdate=func.now(), insert_default=func.now()
        ),
        description="Timestamp when the evaluation task was last updated.",
    )

    progress: float = Field(
        default=0.0,
        description="Progress of the evaluation task as a float between 0.0 and 1.0.",
    )

    evaluation_metrics: str | None = Field(
        default=None,
        description="JSON string containing evaluation metrics.",
    )

    baseline_metrics: str | None = Field(
        default=None,
        description=(
            "JSON string containing baseline metrics (if comparison performed)."
        ),
    )

    evaluation_summary: str | None = Field(
        default=None,
        description="Human-readable evaluation summary.",
    )
