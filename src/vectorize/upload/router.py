"""Router for model upload and management."""

from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from huggingface_hub import model_info
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.ai_model.exceptions import ModelNotFoundError
from vectorize.ai_model.model_source import RemoteModelSource
from vectorize.ai_model.service import get_ai_model_svc
from vectorize.common.exceptions import InternalServerError, InvalidFileError
from vectorize.common.task_status import TaskStatus
from vectorize.config.db import get_session

from .exceptions import InvalidUrlError, ModelAlreadyExistsError
from .exceptions import ModelNotFoundError as RepoModelNotFound
from .github_service import repo_info
from .local_service import upload_zip_model
from .models import UploadTask
from .repository import (
    get_upload_task_by_id_db,
    save_upload_task_db,
)
from .schemas import GitHubModelRequest, HuggingFaceModelRequest
from .tasks import (
    process_github_model_bg,
    process_huggingface_model_bg,
)

router = APIRouter(tags=["AIModel Upload"])


@router.post("/huggingface")
async def load_model_huggingface(
    data: HuggingFaceModelRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Upload a Hugging Face model by model_tag and revision.

    Checks if the model already exists, verifies its presence on
    Hugging Face, creates an upload task, and starts background
    processing. Returns a 201 response with a Location header.

    Args:
        data: Model tag and revision for Hugging Face.
        request: FastAPI request object.
        background_tasks: FastAPI background task manager.
        db: Async database session.

    Returns:
        Response with 201 status and Location header.

    Raises:
        ModelAlreadyExistsError: If the model already exists in the database.
        ModelNotFoundError: If the model is not found on Hugging Face.
        InternalServerError: If an internal error occurs while checking the model.
    """
    key = f"{data.model_tag}@{data.revision}"

    try:
        await get_ai_model_svc(db, key)
        raise ModelAlreadyExistsError(key)
    except ModelNotFoundError:
        pass

    try:
        model_info(repo_id=data.model_tag, revision=data.revision)
    except (EntryNotFoundError, HfHubHTTPError) as e:
        raise ModelNotFoundError(data.model_tag, data.revision) from e
    except Exception as e:
        raise InternalServerError(
            "Internal server error while checking model on Hugging Face.",
        ) from e

    upload_task = UploadTask(
        model_tag=key,
        task_status=TaskStatus.PENDING,
        source=RemoteModelSource.HUGGINGFACE,
    )
    await save_upload_task_db(db, upload_task)

    background_tasks.add_task(
        process_huggingface_model_bg,
        db,
        data.model_tag,
        data.revision,
        upload_task.id,
    )

    return Response(
        status_code=status.HTTP_201_CREATED,
        headers={"Location": f"{request.base_url}v1/upload/tasks/{upload_task.id}"},
    )


@router.post("/github")
async def load_model_github(
    data: GitHubModelRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Upload a Github model by url.

    Checks if the model already exists, verifies its presence in the
    Github Repository, creates an upload task, and starts background
    processing. Returns a 201 response with a Location header.

    Args:
        data: The GitHub Model request Url object
        request: FastAPI request object.
        background_tasks: FastAPI background task manager.
        db: Async database session.

    Returns:
        Response with 201 status and Location header.
    """
    owner = data.owner
    repo = data.repo_name
    branch = data.revision or "main"

    key = f"{owner}/{repo}@{branch}"
    base_url = f"https://github.com/{owner}/{repo}"

    logger.info("Importing GitHub model {} @ {}", repo, branch)

    try:
        await get_ai_model_svc(db, key)
        raise ModelAlreadyExistsError(key)
    except ModelNotFoundError:
        pass

    try:
        repo_info(repo_url=base_url, revision=branch)
    except (RepoModelNotFound, InvalidUrlError):
        raise
    except Exception as e:
        raise InternalServerError("Error checking GitHub repository") from e

    task = UploadTask(
        model_tag=key, task_status=TaskStatus.PENDING, source=RemoteModelSource.GITHUB
    )
    await save_upload_task_db(db, task)
    background_tasks.add_task(
        process_github_model_bg, db, owner, repo, branch, task.id
    )

    return Response(
        status_code=status.HTTP_201_CREATED,
        headers={"Location": f"{request.base_url}v1/upload/tasks/{task.id}"},
    )


@router.post("/local", summary="Upload multiple models from a ZIP archive")
async def load_model_local(
    file: Annotated[UploadFile, File()],
    request: Request,
    db: Annotated[AsyncSession, Depends(get_session)],
    model_name: Annotated[
        str | None, Query(description="Base name for models (optional)")
    ] = None,
) -> Response:
    """Upload a ZIP archive containing multiple model directories.

    Each top-level directory in the ZIP will be treated as a separate model.
    All extracted models will be registered in the database.

    Args:
        file: ZIP archive containing model directories
        request: HTTP request object
        model_name: Optional base name for models (used as fallback)
        db: Database session

    Returns:
        Response with 201 status and Location header

    Raises:
        InvalidFileError: If no file is provided or format is invalid
        HTTPException: If an error occurs during processing
    """
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise InvalidFileError("Only ZIP archives are supported")

    logger.debug(
        "Multi-model ZIP upload: '{}' with {} bytes",
        file.filename,
        file.size if hasattr(file, "size") else "unknown size",
    )

    result = await upload_zip_model(file, model_name, db, multi_model=True)

    model_count = result["total_models"]
    logger.info(f"Successfully uploaded {model_count} models from ZIP archive")

    headers = {}
    if result["models"]:
        first_model = result["models"][0]
        headers["Location"] = f"{request.url}/{first_model['model_id']}"

    return Response(
        status_code=status.HTTP_201_CREATED,
        headers=headers,
    )


@router.get("/{task_id}", summary="Get status of a model upload task")
async def get_upload_status(
    task_id: UUID,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> UploadTask:
    """Endpoint to retrieve the status of an upload task by its ID."""
    task = await get_upload_task_by_id_db(db, task_id)
    if not task:
        raise ModelNotFoundError(f"Upload task with ID {task_id} not found")

    return task
