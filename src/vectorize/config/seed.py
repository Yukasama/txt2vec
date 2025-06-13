"""Seed the database with initial data."""

from datetime import UTC, datetime
from uuid import UUID

from loguru import logger
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.ai_model.model_source import ModelSource, RemoteModelSource
from vectorize.ai_model.models import AIModel
from vectorize.common.task_status import TaskStatus
from vectorize.config.config import settings
from vectorize.dataset.classification import Classification
from vectorize.dataset.dataset_source import DatasetSource
from vectorize.dataset.models import Dataset
from vectorize.upload.models import UploadTask

__all__ = ["seed_db"]


DATASET_READ_ID = UUID("8b8c7f3e-4d2a-4b5c-9f1e-0a6f3e4d2a5b")
DATASET_FAIL_ID = UUID("5d2f3e4b-8c7f-4d2a-9f1e-0a6f3e4d2a5b")
DATASET_PUT_ID = UUID("6d2f3e4b-8c7f-4d2a-9f1e-0a6f3e4d2a5b")
DATASET_DELETE_ID = UUID("7d2f3e4b-8c7f-4d2a-9f1e-0a6f3e4d2a5b")
DATASET_BACKUP_ID = UUID("8d2f3e4b-8c7f-4d2a-9f1e-0a6f3e4d2a5b")
DATASET_BACKUP2_ID = UUID("9d2f3e4b-8c7f-4d2a-9f1e-0a6f3e4d2a5b")

AI_MODEL_READ_ID = UUID("7d2f3e4b-8c7f-4d2a-9f1e-0a6f3e4d2a5b")
AI_MODEL_FAIL_ID = UUID("8d2f3e4b-8c7f-4d2a-9f1e-0a6f3e4d2a5b")
AI_MODEL_DELETE_ID = UUID("2d2f3e4b-8c7f-4d2a-9f1e-0a6f3e4d2a5b")

UPLOAD_TASK_GH_ID = UUID("d2f3e4b8-8c7f-4d2a-9f1e-0a6f3e4d2a5b")
UPLOAD_TASK_HF_ID = UUID("d2f3e4b8-8c7f-4d2a-9f1e-0a6f3e4d2a5c")


async def seed_db(session: AsyncSession) -> None:
    """Seed the database with initial test data.

    Populates the database with example records for development and testing,
    including a sample dataset and AI model.

    Args:
        session: The SQLModel async database session.
    """
    if not settings.clear_db_on_restart:
        statement = select(Dataset)
        result = await session.exec(statement)
        datasets = result.all()
        if datasets:
            return

    session.add(
        Dataset(
            id=DATASET_READ_ID,
            name="read_dataset",
            file_name="read_dataset.jsonl",
            classification=Classification.SENTENCE_DUPLES,
            source=DatasetSource.LOCAL,
            rows=5,
        ),
    )
    session.add(
        Dataset(
            id=DATASET_PUT_ID,
            name="put_dataset",
            file_name="put_dataset.jsonl",
            classification=Classification.SENTENCE_DUPLES,
            source=DatasetSource.LOCAL,
            rows=5,
        ),
    )
    session.add(
        Dataset(
            id=DATASET_DELETE_ID,
            name="delete_dataset",
            file_name="delete_dataset.jsonl",
            classification=Classification.SENTENCE_DUPLES,
            source=DatasetSource.LOCAL,
            rows=5,
        ),
    )
    session.add(
        Dataset(
            id=DATASET_BACKUP_ID,
            name="backup_dataset",
            file_name="backup_dataset.jsonl",
            classification=Classification.SENTENCE_DUPLES,
            source=DatasetSource.LOCAL,
            rows=5,
        ),
    )
    session.add(
        Dataset(
            id=DATASET_BACKUP2_ID,
            name="backup2_dataset",
            file_name="backup2_dataset.jsonl",
            classification=Classification.SENTENCE_DUPLES,
            source=DatasetSource.LOCAL,
            rows=5,
        ),
    )
    session.add(
        Dataset(
            id=DATASET_FAIL_ID,
            name="fail_dataset",
            file_name="fail_dataset.jsonl",
            classification=Classification.SENTENCE_DUPLES,
            source=DatasetSource.LOCAL,
            rows=5,
        ),
    )
    session.add(
        AIModel(
            id=AI_MODEL_READ_ID,
            name="Pytorch Model",
            source=ModelSource.LOCAL,
            model_tag="pytorch_model",
        ),
    )
    session.add(
        AIModel(
            id=AI_MODEL_FAIL_ID,
            name="Big Model",
            source=ModelSource.LOCAL,
            model_tag="big_model",
        ),
    )
    session.add(
        AIModel(
            id=AI_MODEL_DELETE_ID,
            name="Huge Model",
            source=ModelSource.LOCAL,
            model_tag="huge_model",
        ),
    )
    session.add(
        AIModel(
            name="Any Paged Model 01",
            source=ModelSource.LOCAL,
            model_tag="any_model_01",
        ),
    )
    session.add(
        AIModel(
            name="Any Paged Model 02",
            source=ModelSource.LOCAL,
            model_tag="any_model_02",
        ),
    )
    session.add(
        AIModel(
            name="Any Paged Model 03",
            source=ModelSource.LOCAL,
            model_tag="any_model_03",
        ),
    )
    session.add(
        AIModel(
            name="Any Paged Model 04",
            source=ModelSource.LOCAL,
            model_tag="any_model_04",
        ),
    )
    session.add(
        AIModel(
            name="Any Paged Model 05",
            source=ModelSource.LOCAL,
            model_tag="any_model_05",
        ),
    )
    session.add(
        UploadTask(
            id=UPLOAD_TASK_GH_ID,
            model_tag="example-github-model",
            task_status=TaskStatus.PENDING,
            source=RemoteModelSource.GITHUB,
            created_at=datetime(2025, 6, 10, 9, 0, tzinfo=UTC),
            updated_at=datetime(2025, 6, 10, 9, 5, tzinfo=UTC),
            end_date=datetime(2025, 6, 10, 9, 5, tzinfo=UTC),
            error_msg=None,
        ),
    )
    session.add(
        UploadTask(
            id=UPLOAD_TASK_HF_ID,
            model_tag="example-hf-model",
            task_status=TaskStatus.PENDING,
            source=RemoteModelSource.HUGGINGFACE,
            created_at=datetime(2025, 6, 11, 14, 30, tzinfo=UTC),
            updated_at=datetime(2025, 6, 11, 14, 31, tzinfo=UTC),
            end_date=datetime(2025, 6, 11, 14, 35, tzinfo=UTC),
            error_msg=None,
        ),
    )
    await session.commit()
    logger.info("Database seeded with initial data")
