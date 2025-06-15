"""Model path resolution utilities."""

from pathlib import Path

from vectorize.training.utils.safetensors_finder import find_safetensors_file

__all__ = ["resolve_model_path"]


def resolve_model_path(model_tag: str) -> str:
    """Resolve model tag to actual model path.

    Uses recursive search to find the correct model directory structure.
    Works with both HuggingFace cache format and direct model directories.

    Args:
        model_tag: Model tag/path

    Returns:
        str: Resolved model path containing model files

    Raises:
        FileNotFoundError: If model path cannot be resolved
    """
    base_path = Path("data/models") / model_tag

    if not base_path.exists():
        raise FileNotFoundError(f"Model directory not found: {base_path}")

    snapshots_dir = base_path / "snapshots"
    if snapshots_dir.exists():
        snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        if snapshot_dirs:
            candidate_path = snapshot_dirs[0]
            if (candidate_path / "config.json").exists():
                return str(candidate_path)
            safetensors_path = find_safetensors_file(str(candidate_path))
            if safetensors_path:
                return str(candidate_path)

        raise FileNotFoundError(f"No valid model found in snapshots: {snapshots_dir}")

    if (base_path / "config.json").exists():
        return str(base_path)

    safetensors_path = find_safetensors_file(str(base_path))
    if safetensors_path:
        return str(Path(safetensors_path).parent)

    raise FileNotFoundError(f"No valid model files found in: {base_path}")
