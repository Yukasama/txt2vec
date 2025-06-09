"""Schemas for the training API."""

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    """Request body for model training."""
    model_path: str = Field(
        ..., description="Path to the local model directory (no Huggingface download allowed)"
    )
    dataset_paths: list[str] = Field(
        ..., description="Paths to the training datasets (local, multiple allowed)"
    )
    output_dir: str = Field(
        ..., description="Path to save the trained model"
    )
    epochs: int = Field(1, description="Number of training epochs")
    learning_rate: float = Field(5e-5, description="Learning rate for training")
    per_device_train_batch_size: int = Field(8, description="Batch size per device")
