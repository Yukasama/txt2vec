"""Utility to find a .safetensors file in a directory tree."""

import os
from pathlib import Path


def find_safetensors_file(model_dir: str) -> str | None:
    """Recursively search for a .safetensors file in a directory tree.

    Args:
        model_dir (str): The root directory to search.

    Returns:
        Optional[str]: Path to the first .safetensors file found, or None.
    """
    for _root, _dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith(".safetensors"):
                return str(Path(_root) / file)
    return None
