"""Dataset preparation utilities for SBERT training."""

from pathlib import Path

from .input_examples import prepare_input_examples
from .validators import TrainingDataValidator


def prepare_training_data(dataset_paths: list[str]) -> tuple[list, list]:
    """Loads and splits training and validation data from dataset paths.

    Args:
        dataset_paths (list[str]): List of dataset file paths.

    Returns:
        tuple: (train_examples, val_examples)
    """
    df = TrainingDataValidator.validate_dataset(Path(dataset_paths[0]))
    train_examples = prepare_input_examples(df)
    if len(dataset_paths) > 1:
        val_df = TrainingDataValidator.validate_dataset(Path(dataset_paths[1]))
        val_examples = prepare_input_examples(val_df)
    else:
        val_split = int(0.1 * len(train_examples))
        val_examples = train_examples[:val_split]
        train_examples = train_examples[val_split:]
    return train_examples, val_examples
