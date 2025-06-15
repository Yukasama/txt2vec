"""Dataset validation utilities for evaluation."""

from pathlib import Path

import pandas as pd

from vectorize.training.exceptions import DatasetValidationError

__all__ = ["DatasetValidator"]


class DatasetValidator:
    """Handles dataset validation for evaluation."""

    REQUIRED_COLUMNS = {"Question", "Positive", "Negative"}

    @classmethod
    def validate_dataset(cls, dataset_path: Path) -> pd.DataFrame:
        """Validate and load dataset for evaluation.

        Args:
            dataset_path: Path to JSONL dataset file

        Returns:
            Validated DataFrame with required columns

        Raises:
            DatasetValidationError: If dataset is invalid or cannot be loaded
        """
        if not dataset_path.exists():
            raise DatasetValidationError(f"Dataset file does not exist: {dataset_path}")

        if not dataset_path.is_file():
            raise DatasetValidationError(f"Path is not a file: {dataset_path}")

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
                null_count = df[col].isnull().sum()
                raise DatasetValidationError(
                    f"Column '{col}' contains {null_count} null values "
                    f"in {dataset_path}"
                )

        for col in cls.REQUIRED_COLUMNS:
            empty_count = (~df[col].astype(str).str.strip().astype(bool)).sum()
            if empty_count > 0:
                raise DatasetValidationError(
                    f"Column '{col}' contains {empty_count} empty values "
                    f"in {dataset_path}"
                )

        return df
