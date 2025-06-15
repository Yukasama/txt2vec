"""Utility for converting DataFrames to InputExamples for SBERT training."""

import pandas as pd
from sentence_transformers import InputExample
from torch.utils.data import Dataset


class InputExampleDataset(Dataset[InputExample]):
    """Wraps a list of InputExamples for use with DataLoader.

    Args:
        examples: List of InputExample objects.
    """

    def __init__(self, examples: list[InputExample]) -> None:
        """Initializes the dataset.

        Args:
            examples: List of InputExample objects.
        """
        self.examples = examples

    def __getitem__(self, idx: int) -> InputExample:
        """Returns the InputExample at the given index.

        Args:
            idx: Index of the example.

        Returns:
            The example at the given index.
        """
        return self.examples[idx]

    def __len__(self) -> int:
        """Returns the number of examples in the dataset.

        Returns:
            Number of examples.
        """
        return len(self.examples)


def prepare_input_examples(df: pd.DataFrame) -> list[InputExample]:
    """Convert a DataFrame with 'Question', 'Positive', 'Negative' columns.

    Args:
        df: DataFrame with required columns.

    Returns:
        List of InputExample objects for SBERT training.
    """
    examples: list[InputExample] = []
    for _, row in df.iterrows():
        question = str(row["Question"]).strip()
        positive = str(row["Positive"]).strip()
        negative = str(row["Negative"]).strip()

        if not question or not positive or not negative:
            continue

        examples.extend([
            InputExample(texts=[question, positive], label=1.0),
            InputExample(texts=[question, negative], label=-1.0),
        ])
    return examples
    return examples
