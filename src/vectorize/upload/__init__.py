"""Upload module for vectorize."""

from vectorize.upload.exceptions import (
    EmptyModelError,
    InvalidModelError,
    InvalidUrlError,
    InvalidZipError,
    ModelAlreadyExistsError,
    ModelTooLargeError,
    NoValidModelsFoundError,
)
from vectorize.upload.github_service import (
    load_github_model_and_cache_only_svc,
    repo_info,
)
from vectorize.upload.huggingface_service import (
    load_huggingface_model_and_cache_only_svc,
)
from vectorize.upload.local_service import upload_zip_model
from vectorize.upload.models import UploadTask
from vectorize.upload.repository import (
    get_upload_task_by_id_db,
    save_upload_task_db,
    update_upload_task_status_db,
)
from vectorize.upload.router import router
from vectorize.upload.schemas import (
    GitHubModelRequest,
    HuggingFaceModelRequest,
)
from vectorize.upload.tasks import (
    process_github_model_bg,
    process_huggingface_model_bg,
)

__all__ = [
    "EmptyModelError",
    "GitHubModelRequest",
    "HuggingFaceModelRequest",
    "InvalidModelError",
    "InvalidUrlError",
    "InvalidZipError",
    "ModelAlreadyExistsError",
    "ModelTooLargeError",
    "NoValidModelsFoundError",
    "UploadTask",
    "get_upload_task_by_id_db",
    "load_github_model_and_cache_only_svc",
    "load_huggingface_model_and_cache_only_svc",
    "process_github_model_bg",
    "process_huggingface_model_bg",
    "repo_info",
    "router",
    "save_upload_task_db",
    "update_upload_task_status_db",
    "upload_zip_model",
]
