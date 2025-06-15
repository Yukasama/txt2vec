"""Centralized training data validation for SBERT training pipeline."""

from pathlib import Path

import pandas as pd

from ..exceptions import DatasetValidationError


class TrainingDataValidator:
    """Validates training data for SBERT training pipeline.

    Ensures required columns, non-empty data, and no null values.
    """

    REQUIRED_COLUMNS = {"Question", "Positive", "Negative"}

    @classmethod
    def validate_dataset(cls, dataset_path: Path) -> pd.DataFrame:
        """Central validation for all datasets.

        Args:
            dataset_path (Path): Path to the JSONL file.

        Returns:
            pd.DataFrame: Validated DataFrame.

        Raises:
            DatasetValidationError: For invalid or inconsistent data.
        """
        try:
            df = pd.read_json(dataset_path, lines=True)
        except Exception as exc:
            raise DatasetValidationError(
                f"Invalid JSONL file {dataset_path}: {exc}"
            ) from exc

        missing_cols = cls.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise DatasetValidationError(
                f"Missing columns in {dataset_path}: {missing_cols}"
            )

        if df.empty:
            raise DatasetValidationError(f"Dataset {dataset_path} is empty")

        for col in cls.REQUIRED_COLUMNS:
            if bool(df[col].isnull().any()):
                raise DatasetValidationError(
                    f"Column '{col}' contains null values in {dataset_path}"
                )

        return df
