"""Optimized model loader for saved AI-Models with config reuse and fast loading."""

from functools import lru_cache
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    T5EncoderModel,
)

from vectorize.ai_model.exceptions import ModelLoadError, ModelNotFoundError
from vectorize.config import settings

__all__ = ["instantiate_from_weights", "load_model"]


_DEVICE = torch.device(settings.inference_device)
_IS_TORCH_NEW = torch.__version__ >= "2.7"
_MODEL_NAME = "model.bin"

# Optimized loading parameters for better performance
_OPTIMIZED_MODEL_KWARGS = {
    "trust_remote_code": True,
    "torch_dtype": torch.float16 if _DEVICE.type == "cuda" else "auto",
    "low_cpu_mem_usage": True,
    "use_safetensors": True,
    "device_map": _DEVICE if _DEVICE.type == "cuda" else None,
}


@lru_cache(maxsize=10)
def load_model(model_tag: str) -> tuple[torch.nn.Module, AutoTokenizer | None]:
    """Load a Hugging Face model and its tokenizer from a checkpoint directory.

    Optimized version with config reuse and better loading parameters.
    Supports loading from directories containing either a complete HF model snapshot
    or just the config.json and model weights files.

    Args:
        model_tag: Name of the model directory to load (without file extensions).

    Returns:
        A tuple containing:
            - model: The loaded PyTorch model in evaluation mode on target device
            - tokenizer: The model's tokenizer if available, or None if not found

    Raises:
        ModelNotFoundError: If the model directory doesn't exist
        ModelLoadError: If the model can't be successfully loaded
    """
    folder = Path(settings.model_inference_dir) / model_tag
    if not folder.exists():
        logger.error("Model directory not found: {}", folder)
        raise ModelNotFoundError(model_tag)

    logger.info("Loading model '{}' from {}", model_tag, folder)

    try:
        # Load config once and reuse it
        logger.debug("Loading config from {}", folder / "config.json")
        config = AutoConfig.from_pretrained(folder)

        # Log config details for debugging
        model_type = getattr(config, "model_type", "unknown")
        architectures = getattr(config, "architectures", [])
        logger.info(
            "Config loaded - model_type: '{}', architectures: {}",
            model_type,
            architectures,
        )

        # Use AutoModel with config reuse
        logger.debug("Attempting to load model with AutoModel.from_pretrained")
        model = AutoModel.from_pretrained(
            folder, config=config, **_OPTIMIZED_MODEL_KWARGS
        )
        logger.info(
            "Model loaded successfully with AutoModel: {}", model.__class__.__name__
        )

    except OSError as ose_exc:
        # Fallback for incomplete models (missing standard files)
        logger.warning(
            "AutoModel loading failed with OSError: {}. Trying fallback method.",
            str(ose_exc),
        )
        try:
            config = AutoConfig.from_pretrained(folder)
            model = instantiate_from_weights(folder, config)
            logger.info("Model loaded successfully with fallback method")
        except Exception as fallback_exc:
            logger.error("Fallback loading also failed: {}", str(fallback_exc))
            raise ModelLoadError(model_tag) from fallback_exc
    except Exception as exc:
        logger.error("Model loading failed: {}", str(exc))
        raise ModelLoadError(model_tag) from exc

    # Only transfer to device if needed
    if (hasattr(model, "device") and model.device != _DEVICE) or not hasattr(
        model, "device"
    ):
        logger.debug("Moving model to device: {}", _DEVICE)
        model = model.to(_DEVICE)
    else:
        logger.debug("Model already on correct device: {}", _DEVICE)

    model = model.eval().requires_grad_(False)
    logger.debug("Model set to eval mode with requires_grad=False")

    # Load tokenizer
    try:
        logger.debug("Loading tokenizer from {}", folder)
        tok = AutoTokenizer.from_pretrained(folder, trust_remote_code=True)
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.warning("No tokenizer available for '{}': {}", model_tag, e)
        tok = None

    logger.info(
        "Model loading complete - {} on {}, tokenizer: {}",
        model.__class__.__name__,
        _DEVICE,
        "available" if tok is not None else "not available",
    )
    return model, tok


def instantiate_from_weights(folder: Path, cfg: AutoConfig) -> torch.nn.Module:
    """Create a model from config.json and load weights from a state dict file.

    This is a fallback method for loading models that don't follow the standard
    Hugging Face directory structure but contain the necessary components.

    Args:
        folder: Path to the directory containing config.json and weight files.
        cfg: The AutoConfig object containing model configuration.

    Returns:
        The loaded PyTorch model in evaluation mode.

    Raises:
        ModelNotFoundError: If neither pytorch_model.bin nor model.bin exist
    """
    logger.info("Using fallback: instantiating model from weights manually")

    # Extract config information safely
    model_type = getattr(cfg, "model_type", None)
    architectures = getattr(cfg, "architectures", None)
    config_dict = cfg.to_dict()  # type: ignore

    logger.debug("  model_type: {}", model_type)
    logger.debug("  architectures: {}", architectures)
    logger.debug("  config keys: {}", list(config_dict.keys()))

    # Create model architecture from config
    if model_type == "t5":
        logger.info("T5 model detected - using T5EncoderModel")
        model = T5EncoderModel.from_config(cfg)  # type: ignore
        model_class_name = "T5EncoderModel"
    elif architectures and len(architectures) > 0 and "MaskedLM" in architectures[0]:
        logger.debug("  Architecture: {}", architectures[0])
        model = AutoModelForMaskedLM.from_config(cfg)
        model_class_name = "AutoModelForMaskedLM"
    else:
        logger.info("Standard model detected - using AutoModel")
        if architectures:
            logger.debug(
                "  Architecture: {}", architectures[0] if architectures else "None"
            )
        model = AutoModel.from_config(cfg)
        model_class_name = "AutoModel"

    logger.info("Model architecture created: {}", model_class_name)

    # Move to device
    logger.debug("Moving model to device: {}", _DEVICE)
    model.to(_DEVICE)

    # Try to load weights from safetensors first (faster and safer)
    logger.debug("Searching for weight files in {}", folder)
    safetensors_files = list(folder.glob("*.safetensors"))

    if safetensors_files:
        st_path = safetensors_files[0]
        logger.info("Loading weights from safetensors: {}", st_path.name)
        if len(safetensors_files) > 1:
            logger.warning("Multiple safetensors files found, using: {}", st_path.name)

        try:
            state = load_file(st_path, device=str(_DEVICE))
            model.load_state_dict(state, strict=False)
            logger.info(
                "Safetensors weights loaded successfully ({} parameters)", len(state)
            )
            return model.eval()
        except Exception as e:
            logger.error("Failed to load safetensors: {}", str(e))
            # Continue to try pytorch format

    # Fallback to pytorch binary format
    pytorch_file = folder / _MODEL_NAME
    if pytorch_file.is_file():
        logger.info("Loading weights from PyTorch binary: {}", pytorch_file.name)
        try:
            state = torch.load(
                pytorch_file,
                mmap=_IS_TORCH_NEW,
                map_location=_DEVICE,
                weights_only=_IS_TORCH_NEW,
            )
            model.load_state_dict(state, strict=False)
            logger.info("PyTorch binary weights loaded successfully")
            return model.eval()
        except Exception as e:
            logger.error("Failed to load PyTorch binary: {}", str(e))

    # No weight files found
    available_files = [f.name for f in folder.iterdir() if f.is_file()]
    logger.error(
        "No weight files found in {}. Available files: {}", folder, available_files
    )
    raise ModelNotFoundError(str(folder))
