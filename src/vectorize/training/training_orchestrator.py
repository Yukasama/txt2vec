"""Orchestrates the end-to-end SBERT training process."""

from pathlib import Path
from uuid import UUID

from loguru import logger
from sentence_transformers import SentenceTransformer, losses
from sqlmodel.ext.asyncio.session import AsyncSession
from torch.utils.data import DataLoader

from vectorize.ai_model.model_source import ModelSource
from vectorize.ai_model.models import AIModel
from vectorize.ai_model.repository import get_ai_model_db, save_ai_model_db
from vectorize.common.task_status import TaskStatus

from .exceptions import DatasetValidationError
from .repository import get_train_task_by_id, update_training_task_status
from .schemas import TrainRequest
from .utils.cleanup import cleanup_resources
from .utils.input_examples import InputExampleDataset, prepare_input_examples
from .utils.model_loader import load_and_prepare_model
from .utils.validators import TrainingDataValidator


class TrainingOrchestrator:
    """Handles the orchestration of SBERT training."""

    def __init__(self, db: AsyncSession, task_id: UUID) -> None:
        """Initialize the training orchestrator.

        Args:
            db: Database session
            task_id: Training task ID
        """
        self.db = db
        self.task_id = task_id
        self.model: SentenceTransformer | None = None

    async def run_training(
        self,
        model_path: str,
        train_request: TrainRequest,
        dataset_paths: list[str],
        output_dir: str,
    ) -> None:
        """Main orchestration function for SBERT training.

        Args:
            model_path: Path to the base model
            train_request: Training configuration
            dataset_paths: List of dataset file paths
            output_dir: Output directory for the trained model
        """
        logger.debug(
            "Starting SBERT training",
            model_path=model_path,
            dataset_paths=dataset_paths,
            output_dir=output_dir,
            task_id=str(self.task_id),
        )

        try:
            self.model = load_and_prepare_model(model_path)

            train_dataloader = TrainingOrchestrator._prepare_training_data(
                dataset_paths, train_request.per_device_train_batch_size
            )

            self._train_model(train_dataloader, train_request, output_dir)

            await self._save_trained_model(train_request, output_dir)

            await update_training_task_status(self.db, self.task_id, TaskStatus.DONE)
            logger.debug(
                "Training finished successfully",
                model_path=model_path,
                task_id=str(self.task_id),
            )

        except (OSError, RuntimeError, DatasetValidationError) as exc:
            await self._handle_training_error(exc)
        finally:
            self._cleanup()

    @staticmethod
    def _prepare_training_data(dataset_paths: list[str], batch_size: int) -> DataLoader:
        """Prepare training data from dataset paths.

        Args:
            dataset_paths: List of dataset file paths
            batch_size: Training batch size

        Returns:
            DataLoader for training
        """
        df = TrainingDataValidator.validate_dataset(Path(dataset_paths[0]))
        train_examples = prepare_input_examples(df)

        if len(dataset_paths) > 1:
            val_df = TrainingDataValidator.validate_dataset(Path(dataset_paths[1]))
            _ = prepare_input_examples(val_df)
        else:
            val_split = int(0.1 * len(train_examples))
            train_examples = train_examples[val_split:]

        train_dataset = InputExampleDataset(train_examples)
        return DataLoader(train_dataset, batch_size=batch_size)

    def _train_model(
        self,
        train_dataloader: DataLoader,
        train_request: TrainRequest,
        output_dir: str,
    ) -> None:
        """Train the SBERT model.

        Args:
            train_dataloader: Training data loader
            train_request: Training configuration
            output_dir: Output directory for the trained model
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        loss = losses.CosineSimilarityLoss(self.model)

        self.model.fit(
            train_objectives=[(train_dataloader, loss)],
            epochs=train_request.epochs,
            warmup_steps=train_request.warmup_steps or 0,
            show_progress_bar=False,
            output_path=str(Path(output_dir)),
        )

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save(str(output_dir))

    async def _save_trained_model(
        self, train_request: TrainRequest, output_dir: str
    ) -> None:
        """Save the trained model to the database.

        Args:
            train_request: Training configuration
            output_dir: Output directory where model was saved
        """
        parent_model = await get_ai_model_db(self.db, train_request.model_tag)
        tag_time = Path(output_dir).name
        new_model_tag = str(Path(output_dir).relative_to("data/models"))

        new_model = AIModel(
            name=f"Fine-tuned: {parent_model.name} {tag_time}",
            model_tag=new_model_tag,
            source=ModelSource.LOCAL,
            trained_from_id=parent_model.id,
            trained_from_tag=parent_model.model_tag,
        )

        new_model_id = await save_ai_model_db(self.db, new_model)

        task = await get_train_task_by_id(self.db, self.task_id)
        if task:
            task.trained_model_id = new_model_id
            await self.db.commit()
            await self.db.refresh(task)

    async def _handle_training_error(self, exc: Exception) -> None:
        """Handle training errors.

        Args:
            exc: The exception that occurred
        """
        logger.error(
            "Training failed",
            task_id=str(self.task_id),
            exc=str(exc),
        )
        await update_training_task_status(
            self.db,
            self.task_id,
            TaskStatus.FAILED,
            error_msg=str(exc),
        )

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self.model is not None:
            cleanup_resources(self.model)
            self.model = None


async def run_training(  # noqa: PLR0913, PLR0917
    db: AsyncSession,
    model_path: str,
    train_request: TrainRequest,
    task_id: UUID,
    dataset_paths: list[str],
    output_dir: str,
) -> None:
    """Legacy wrapper function for backward compatibility.

    Args:
        db: Database session
        model_path: Path to the base model
        train_request: Training configuration
        task_id: Training task ID
        dataset_paths: List of dataset file paths
        output_dir: Output directory for the trained model
    """
    orchestrator = TrainingOrchestrator(db, task_id)
    await orchestrator.run_training(
        model_path, train_request, dataset_paths, output_dir
    )
