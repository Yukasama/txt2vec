"""Model loading and tokenizer preparation utilities for SBERT training."""

from pathlib import Path

from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers.tokenization_utils import PreTrainedTokenizer

from .safetensors_finder import find_safetensors_file


def load_and_prepare_model(model_path: str) -> SentenceTransformer:
    """Loads a SentenceTransformer model and prepares its tokenizer.

    Args:
        model_path (str): Path to the base model directory.

    Returns:
        SentenceTransformer: The loaded and prepared model.
    """
    safetensors_path = find_safetensors_file(model_path)
    if safetensors_path:
        model_dir = Path(safetensors_path).parent
        logger.debug(
            "Found .safetensors file for",
            safetensors_path=safetensors_path,
            model_dir=model_dir,
        )
        model = SentenceTransformer(str(model_dir))
    else:
        logger.debug("No .safetensors file found, loading model from original path.")
        model = SentenceTransformer(model_path)
    _prepare_tokenizer(model.tokenizer)
    return model


def _prepare_tokenizer(tokenizer: PreTrainedTokenizer) -> None:
    """Ensures the tokenizer has a pad token set.

    Args:
        tokenizer: The tokenizer object to prepare.
    """
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.debug("Set pad_token to eos_token", eos_token=tokenizer.eos_token)
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            logger.debug("Added [PAD] token as pad_token")
    else:
        # Escape the pad_token to avoid Loguru color parsing issues
        pad_token_str = str(tokenizer.pad_token).replace("<", "\\<").replace(">", "\\>")
        logger.debug("Tokenizer already has pad_token", pad_token=pad_token_str)
