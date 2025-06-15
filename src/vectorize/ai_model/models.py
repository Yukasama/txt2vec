"""AIModel models."""

from datetime import datetime
from typing import TYPE_CHECKING, Optional
from uuid import UUID, uuid4

from sqlmodel import Column, DateTime, Field, Relationship, SQLModel, func

from .model_source import ModelSource

if TYPE_CHECKING:
    from vectorize.evaluation.models import EvaluationTask
    from vectorize.inference.models import InferenceCounter
    from vectorize.training.models import TrainingTask

__all__ = ["AIModel", "AIModelAll", "AIModelCreate", "AIModelPublic", "AIModelUpdate"]


class _AIModelBase(SQLModel):
    """Base AIModel model."""

    name: str = Field(description="Name of the AI model")

    model_tag: str = Field(description="Tag of the AI model file.")

    source: ModelSource = Field(
        description="Source of the model (github, huggingface, or local)."
    )


class AIModelCreate(_AIModelBase):
    """AIModel creation model."""


class AIModelUpdate(SQLModel):
    """AIModel update model with optional fields."""

    name: str = Field(
        description="Name of the AI model to update", min_length=1, max_length=128
    )


class AIModelAll(_AIModelBase):
    """AIModel model for listing with limited fields."""

    id: UUID = Field(description="Unique identifier for the AI model")

    version: int = Field(description="Version number of the AI model")

    created_at: datetime = Field(description="Timestamp when the AI model was created")


class AIModelPublic(AIModelAll):
    """AIModel model for detailed view with all fields."""

    updated_at: datetime = Field(
        description="Timestamp when the AI model was last updated"
    )


class AIModel(SQLModel, table=True):
    """AIModel model."""

    __tablename__ = "ai_model"

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        description="Unique identifier for the AI model.",
    )

    version: int = Field(default=0, description="Version number of the AI model.")

    name: str = Field(description="Name of the AI model.")

    model_tag: str = Field(
        index=True, unique=True, description="Tag of the AI model file."
    )

    source: ModelSource = Field(
        description="Source of the model (github, huggingface, or local)."
    )

    inference_counters: list["InferenceCounter"] = Relationship(
        back_populates="ai_model",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )

    trained_from_id: UUID | None = Field(
        default=None,
        foreign_key="ai_model.id",
        description="ID of the model this was trained from",
    )

    trained_from: Optional["AIModel"] = Relationship(
        back_populates="trained_children",
        sa_relationship_kwargs={"remote_side": "AIModel.id"},
    )

    trained_children: list["AIModel"] = Relationship(back_populates="trained_from")

    training_tasks: list["TrainingTask"] = Relationship(back_populates="trained_model")

    evaluation_tasks: list["EvaluationTask"] = Relationship(back_populates="model")

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), insert_default=func.now()),
        description="Timestamp when the AI model was created.",
    )

    updated_at: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True), onupdate=func.now(), insert_default=func.now()
        ),
        description="Timestamp when the AI model was last updated.",
    )

    trained_from_tag: str | None = Field(
        default=None,
        description="Tag of the parent model this model was trained from.",
    )
