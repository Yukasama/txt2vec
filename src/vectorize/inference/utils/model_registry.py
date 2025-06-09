"""Model registry for optimized embedding extraction strategies."""

from collections.abc import Callable
from typing import Any

import torch
from loguru import logger

from .pool_mean import _mean_pool

__all__ = ["get_extraction_strategy"]


def _bert_strategy(model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
    """Optimal strategy for BERT-like models."""
    if (
        hasattr(model_output, "pooler_output")
        and model_output.pooler_output is not None
    ):
        return model_output.pooler_output.squeeze(0)
    if hasattr(model_output, "last_hidden_state"):
        return _mean_pool(model_output.last_hidden_state, attention_mask).squeeze(0)
    raise ValueError("BERT model output missing expected attributes")


def _sentence_bert_strategy(
    model_output: Any, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Strategy for Sentence-BERT models (direct tensor)."""
    if isinstance(model_output, torch.Tensor):
        return model_output.squeeze(0)
    if hasattr(model_output, "sentence_embedding"):
        return model_output.sentence_embedding.squeeze(0)
    return _bert_strategy(model_output, attention_mask)


def _t5_strategy(model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
    """Strategy for T5 models (only last_hidden_state)."""
    if hasattr(model_output, "last_hidden_state"):
        return _mean_pool(model_output.last_hidden_state, attention_mask).squeeze(0)
    raise ValueError("T5 model output missing last_hidden_state")


def _distilbert_strategy(
    model_output: Any, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Strategy for DistilBERT (no pooler_output)."""
    if hasattr(model_output, "last_hidden_state"):
        return _mean_pool(model_output.last_hidden_state, attention_mask).squeeze(0)
    raise ValueError("DistilBERT model output missing last_hidden_state")


def _gpt_strategy(model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
    """Strategy for GPT models."""
    if hasattr(model_output, "logits"):
        return model_output.logits.mean(dim=1).squeeze(0)
    if hasattr(model_output, "last_hidden_state"):
        return _mean_pool(model_output.last_hidden_state, attention_mask).squeeze(0)
    raise ValueError("GPT model output missing expected attributes")


EXACT_MODEL_STRATEGIES: dict[str, Callable[[Any, torch.Tensor], torch.Tensor]] = {
    "bert-base-german": _bert_strategy,
    "bert-large-german": _bert_strategy,
    "bert-base-uncased": _bert_strategy,
    "bert-base-cased": _bert_strategy,
    "bert-large-uncased": _bert_strategy,
    "roberta-base": _bert_strategy,
    "roberta-large": _bert_strategy,
    "distilbert-base-german": _distilbert_strategy,
    "distilbert-base-uncased": _distilbert_strategy,
    "distilbert-base-cased": _distilbert_strategy,
    "sentence-bert": _sentence_bert_strategy,
    "all-MiniLM-L6-v2": _sentence_bert_strategy,
    "all-mpnet-base-v2": _sentence_bert_strategy,
    "paraphrase-MiniLM-L6-v2": _sentence_bert_strategy,
    "t5-base": _t5_strategy,
    "t5-large": _t5_strategy,
    "t5-small": _t5_strategy,
    "pytorch_model": _bert_strategy,
    "big_model": _bert_strategy,
    "huge_model": _bert_strategy,
}

PATTERN_STRATEGIES: dict[str, Callable[[Any, torch.Tensor], torch.Tensor]] = {
    "bert": _bert_strategy,
    "roberta": _bert_strategy,
    "distilbert": _distilbert_strategy,
    "sentence": _sentence_bert_strategy,
    "miniLM": _sentence_bert_strategy,
    "mpnet": _sentence_bert_strategy,
    "t5": _t5_strategy,
    "gpt": _gpt_strategy,
    "gpt2": _gpt_strategy,
}


def get_extraction_strategy(
    model_tag: str,
) -> Callable[[Any, torch.Tensor], torch.Tensor] | None:
    """Determine optimal embedding extraction strategy for a model.

    Args:
        model_tag: Tag/name of the model

    Returns:
        Strategy function that accepts (model_output, attention_mask)
        or None if no specific strategy found
    """
    if model_tag in EXACT_MODEL_STRATEGIES:
        logger.debug(f"Using exact strategy for model '{model_tag}'")
        return EXACT_MODEL_STRATEGIES[model_tag]

    model_tag_lower = model_tag.lower()
    for pattern, strategy in PATTERN_STRATEGIES.items():
        if pattern.lower() in model_tag_lower:
            logger.debug(f"Using pattern '{pattern}' strategy for model '{model_tag}'")
            return strategy

    logger.debug(
        f"No specific strategy found for model '{model_tag}', will use fallback"
    )
    return None
