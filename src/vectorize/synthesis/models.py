"""Synthesis Task model."""

from datetime import datetime
from typing import TYPE_CHECKING, Optional
from uuid import UUID, uuid4

from sqlmodel import Column, DateTime, Field, Relationship, SQLModel, func

from vectorize.common.task_status import TaskStatus

if TYPE_CHECKING:
    from vectorize.dataset.models import Dataset

__all__ = ["SynthesisTask"]


class SynthesisTask(SQLModel, table=True):
    """Synthetic generation model."""

    __tablename__ = "synthesis_task"

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        description="Unique identifier for the synthetic generation.",
    )

    task_status: TaskStatus = Field(
        default=TaskStatus.QUEUED,
        index=True,
        description="Current status of the synthetic generation.",
    )

    end_date: datetime | None = Field(
        default=None, description="Optional end time of the synthetic generation."
    )

    error_msg: str | None = Field(
        default=None,
        description="Optional error message encountered during generation.",
    )

    generated_dataset: Optional["Dataset"] = Relationship(
        back_populates="synthesis_task"
    )

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), insert_default=func.now()),
        description="Timestamp when the synthesis generation was created.",
    )

    updated_at: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True), onupdate=func.now(), insert_default=func.now()
        ),
        description="Timestamp when the synthesis generation was last updated.",
    )
