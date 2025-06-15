"""UUID validation utility for training module."""

import uuid


def is_valid_uuid(val: str) -> bool:
    """Checks if the given value is a valid UUID."""
    try:
        uuid.UUID(str(val))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def normalize_uuid(val: str) -> str:
    """Returns the UUID as a 32-character string without dashes."""
    try:
        return uuid.UUID(str(val)).hex
    except (ValueError, AttributeError, TypeError):
        return val
