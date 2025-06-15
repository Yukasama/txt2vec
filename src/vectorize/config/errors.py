"""Text constants for consistent error handling."""

from enum import StrEnum

__all__ = ["ErrorCode", "ErrorNames"]


class ErrorCode(StrEnum):
    """Error codes for standardized error handling."""

    # General errors
    SERVER_ERROR = "SERVER_ERROR"
    NOT_FOUND = "NOT_FOUND"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    VERSION_MISMATCH = "VERSION_MISMATCH"
    VERSION_MISSING = "VERSION_MISSING"
    MODEL_ALREADY_EXISTS = "MODEL_ALREADY_EXISTS"

    # Dataset errors
    INVALID_FILE = "INVALID_FILE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    INVALID_CSV_FORMAT = "INVALID_CSV_FORMAT"
    EMPTY_FILE = "EMPTY_FILE"
    HUGGINGFACE_DATASET_FORMAT_ERROR = "HUGGINGFACE_DATASET_FORMAT_ERROR"
    DATASET_ALREADY_EXISTS = "HUGGINGFACE_DATASET_ALREADY_EXISTS"

    # training errors
    INVALID_EPOCHS = "INVALID_EPOCHS"
    INVALID_BATCH_SIZE = "INVALID_BATCH_SIZE"
    INVALID_LEARNING_RATE = "INVALID_LEARNING_RATE"

    INVALID_URL = "INVALID_URL"


class ErrorNames(StrEnum):
    """Error names for standardized error handling."""

    INTERNAL_SERVER_ERROR = "Internal server error"

    # File validation errors
    FILE_MISSING_ERROR = "No file provided"
    FILENAME_MISSING_ERROR = "File has no filename"
    FILENAME_TOO_LONG_ERROR = "Filename is too long"

    # Format errors
    FORMAT_INVALID_CSV_ERROR = "CSV is malformed near {value}"
    DETECT_MALICIOUS_CONTENT = "Detected invalid formulas or malicious in CSV file"
    DOUBLE_QUOTES_ERROR = "Double-quotes within value must be properly escaped"
    SPECIAL_CHARS_ERROR = "Special characters must be properly quoted"
    INVALID_COLUMN_TYPE = "Column name must be a string"
