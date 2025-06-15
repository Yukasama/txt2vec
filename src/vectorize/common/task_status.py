"""TaskStatus model for task or upload."""

from enum import StrEnum

__all__ = ["TaskStatus"]


class TaskStatus(StrEnum):
    """Status of a process."""

    QUEUED = "Q"
    RUNNING = "R"
    DONE = "D"
    FAILED = "F"
    CANCELED = "C"
