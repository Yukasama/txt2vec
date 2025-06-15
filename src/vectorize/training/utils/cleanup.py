"""Resource cleanup utilities for SBERT training."""

import gc
from typing import TYPE_CHECKING

import torch
from loguru import logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def cleanup_resources(model: "SentenceTransformer | None" = None) -> None:
    """Cleans up model and frees memory resources.

    Args:
        model: The model object to delete (optional).
    """
    if model is not None:
        try:
            del model
            logger.debug("Model successfully deleted from memory")
        except Exception as exc:
            logger.warning("Cleanup failed (model deletion)", error=str(exc))

    try:
        collected = gc.collect()
        logger.debug("Garbage collection completed", objects_collected=collected)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("CUDA cache cleared and synchronized")

    except Exception as exc:
        logger.warning("Cleanup failed (GC/CUDA)", error=str(exc))
