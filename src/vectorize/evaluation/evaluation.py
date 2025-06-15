"""Evaluation utilities for SBERT training quality assessment.

Computes cosine similarity metrics between question-positive-negative triplets
to assess training effectiveness. Main metrics:
- Average cosine similarity between question and positive examples
- Average cosine similarity between question and negative examples
- Ratio of positive to negative similarities (should be > 1)
- Spearman correlation for similarity ranking
"""

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from .utils import DatasetValidator, SimilarityCalculator

__all__ = ["EvaluationMetrics", "TrainingEvaluator"]

TRAINING_SUCCESS_THRESHOLD = 1.2
DEFAULT_MAX_SAMPLES = 1000
DEFAULT_RANDOM_SEED = 42

EXCELLENT_RATIO_THRESHOLD = 2.0
EXCELLENT_CORRELATION_THRESHOLD = 0.7
GOOD_RATIO_THRESHOLD = 1.5
GOOD_CORRELATION_THRESHOLD = 0.5
FAIR_RATIO_THRESHOLD = 1.2
FAIR_CORRELATION_THRESHOLD = 0.3


class EvaluationMetrics:
    """Container for training evaluation metrics."""

    def __init__(
        self,
        avg_positive_similarity: float,
        avg_negative_similarity: float,
        similarity_ratio: float,
        spearman_correlation: float,
        num_samples: int,
    ) -> None:
        """Initialize evaluation metrics.

        Args:
            avg_positive_similarity: Average cosine similarity (question, positive)
            avg_negative_similarity: Average cosine similarity (question, negative)
            similarity_ratio: Ratio of positive to negative similarities
            spearman_correlation: Spearman correlation coefficient
            num_samples: Number of triplets evaluated

        Raises:
            ValueError: If any metric values are invalid
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        if not (-1.0 <= avg_positive_similarity <= 1.0):
            raise ValueError(
                "avg_positive_similarity must be in [-1, 1], "
                f"got {avg_positive_similarity}"
            )

        if not (-1.0 <= avg_negative_similarity <= 1.0):
            raise ValueError(
                "avg_negative_similarity must be in [-1, 1], "
                f"got {avg_negative_similarity}"
            )

        if similarity_ratio < 0:
            raise ValueError(
                f"similarity_ratio must be non-negative, got {similarity_ratio}"
            )

        if not (-1.0 <= spearman_correlation <= 1.0):
            raise ValueError(
                f"spearman_correlation must be in [-1, 1], got {spearman_correlation}"
            )

        self.avg_positive_similarity = avg_positive_similarity
        self.avg_negative_similarity = avg_negative_similarity
        self.similarity_ratio = similarity_ratio
        self.spearman_correlation = spearman_correlation
        self.num_samples = num_samples

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary format.

        Returns:
            Dictionary containing all metrics and training success status
        """
        return {
            "avg_positive_similarity": self.avg_positive_similarity,
            "avg_negative_similarity": self.avg_negative_similarity,
            "similarity_ratio": self.similarity_ratio,
            "spearman_correlation": self.spearman_correlation,
            "num_samples": self.num_samples,
            "is_training_successful": self.is_training_successful(),
            "quality_grade": self.get_quality_grade(),
        }

    def to_baseline_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary format for baseline models.

        Returns:
            Dictionary containing metrics without training-specific fields
        """
        return {
            "avg_positive_similarity": self.avg_positive_similarity,
            "avg_negative_similarity": self.avg_negative_similarity,
            "similarity_ratio": self.similarity_ratio,
            "spearman_correlation": self.spearman_correlation,
            "num_samples": self.num_samples,
            "quality_grade": self.get_quality_grade(),
        }

    def is_training_successful(self) -> bool:
        """Simple heuristic to determine if training was successful.

        Returns:
            True if positive similarities > negative similarities and ratio > 1.2
        """
        return (
            self.avg_positive_similarity > self.avg_negative_similarity
            and self.similarity_ratio > TRAINING_SUCCESS_THRESHOLD
        )

    def get_improvement_over_baseline(
        self, baseline: "EvaluationMetrics"
    ) -> dict[str, float]:
        """Compare this model's metrics against a baseline.

        Args:
            baseline: Baseline metrics to compare against

        Returns:
            Dictionary with improvement metrics
        """
        return {
            "positive_similarity_improvement": (
                self.avg_positive_similarity - baseline.avg_positive_similarity
            ),
            "negative_similarity_improvement": (
                self.avg_negative_similarity - baseline.avg_negative_similarity
            ),
            "ratio_improvement": self.similarity_ratio - baseline.similarity_ratio,
            "correlation_improvement": (
                self.spearman_correlation - baseline.spearman_correlation
            ),
        }

    def get_quality_grade(self) -> str:
        """Get a qualitative grade for the training quality.

        Returns:
            Quality grade: "Excellent", "Good", "Fair", "Poor"
        """
        if (
            self.similarity_ratio >= EXCELLENT_RATIO_THRESHOLD
            and self.spearman_correlation >= EXCELLENT_CORRELATION_THRESHOLD
        ):
            return "Excellent"
        if (
            self.similarity_ratio >= GOOD_RATIO_THRESHOLD
            and self.spearman_correlation >= GOOD_CORRELATION_THRESHOLD
        ):
            return "Good"
        if (
            self.similarity_ratio >= FAIR_RATIO_THRESHOLD
            and self.spearman_correlation >= FAIR_CORRELATION_THRESHOLD
        ):
            return "Fair"
        return "Poor"

    def __str__(self) -> str:
        """String representation of metrics."""
        return (
            f"EvaluationMetrics(\n"
            f"  avg_positive_similarity={self.avg_positive_similarity:.4f}\n"
            f"  avg_negative_similarity={self.avg_negative_similarity:.4f}\n"
            f"  similarity_ratio={self.similarity_ratio:.4f}\n"
            f"  spearman_correlation={self.spearman_correlation:.4f}\n"
            f"  num_samples={self.num_samples}\n"
            f"  training_successful={self.is_training_successful()}\n"
            f")"
        )

    def baseline_str(self) -> str:
        """String representation for baseline metrics (without training_successful)."""
        return (
            f"BaselineMetrics(\n"
            f"  avg_positive_similarity={self.avg_positive_similarity:.4f}\n"
            f"  avg_negative_similarity={self.avg_negative_similarity:.4f}\n"
            f"  similarity_ratio={self.similarity_ratio:.4f}\n"
            f"  spearman_correlation={self.spearman_correlation:.4f}\n"
            f"  num_samples={self.num_samples}\n"
            f")"
        )


class TrainingEvaluator:
    """Evaluates SBERT training quality using cosine similarity metrics."""

    def __init__(self, model_path: str) -> None:
        """Initialize evaluator with trained model.

        Args:
            model_path: Path to the trained SentenceTransformer model
        """
        self.model_path = model_path
        self.model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        """Load the SentenceTransformer model.

        Returns:
            Loaded SentenceTransformer model

        Raises:
            FileNotFoundError: If model path does not exist
            Exception: If model loading fails
        """
        if self.model is None:
            logger.debug("Loading model for evaluation", model_path=self.model_path)
            try:
                self.model = SentenceTransformer(self.model_path)
                logger.debug("Model loaded successfully", model_path=self.model_path)
            except Exception as exc:
                logger.error(
                    "Failed to load model", model_path=self.model_path, error=str(exc)
                )
                raise Exception(
                    f"Failed to load model from {self.model_path}: {exc}"
                ) from exc
        return self.model

    def evaluate_dataset(
        self,
        dataset_path: Path,
        max_samples: int | None = None,
        log_results: bool = True,
    ) -> EvaluationMetrics:
        """Evaluate training quality on a dataset.

        Args:
            dataset_path: Path to evaluation dataset (JSONL)
            max_samples: Optional limit on number of samples to evaluate
            log_results: Whether to log the evaluation results

        Returns:
            EvaluationMetrics with computed similarity metrics

        Raises:
            DatasetValidationError: If dataset is invalid
        """
        logger.debug("Starting evaluation", dataset_path=str(dataset_path))

        df = DatasetValidator.validate_dataset(dataset_path)

        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=DEFAULT_RANDOM_SEED)
            logger.debug("Limited evaluation to max samples", max_samples=max_samples)

        model = self._load_model()

        questions = df["Question"].tolist()
        positives = df["Positive"].tolist()
        negatives = df["Negative"].tolist()

        logger.debug("Computing embeddings for triplets", num_triplets=len(questions))

        positive_similarities, negative_similarities = (
            SimilarityCalculator.compute_cosine_similarities(
                model, questions, positives, negatives
            )
        )

        pos_sims = np.array(positive_similarities)
        neg_sims = np.array(negative_similarities)

        avg_positive_similarity = float(np.mean(pos_sims))
        avg_negative_similarity = float(np.mean(neg_sims))

        similarity_ratio = SimilarityCalculator.compute_similarity_ratio(
            avg_positive_similarity, avg_negative_similarity
        )

        spearman_corr = SimilarityCalculator.compute_spearman_correlation(
            positive_similarities, negative_similarities
        )

        metrics = EvaluationMetrics(
            avg_positive_similarity=avg_positive_similarity,
            avg_negative_similarity=avg_negative_similarity,
            similarity_ratio=similarity_ratio,
            spearman_correlation=spearman_corr,
            num_samples=len(questions),
        )

        if log_results:
            logger.debug("Evaluation completed", metrics=str(metrics))
        return metrics

    def compare_models(
        self,
        baseline_model_path: str,
        dataset_path: Path,
        max_samples: int | None = None,
    ) -> dict[str, EvaluationMetrics]:
        """Compare trained model against baseline model.

        Args:
            baseline_model_path: Path to baseline model for comparison
            dataset_path: Path to evaluation dataset
            max_samples: Optional limit on samples

        Returns:
            Dictionary with 'trained' and 'baseline' metrics
        """
        logger.debug(
            "Comparing models", trained=self.model_path, baseline=baseline_model_path
        )

        logger.debug("Evaluating baseline model")
        baseline_evaluator = TrainingEvaluator(baseline_model_path)
        baseline_metrics = baseline_evaluator.evaluate_dataset(
            dataset_path, max_samples, log_results=False
        )

        logger.debug("Evaluating trained model")
        trained_metrics = self.evaluate_dataset(
            dataset_path, max_samples, log_results=False
        )

        logger.debug(
            "Baseline model evaluation",
            model_type="baseline",
            model_path=baseline_model_path,
            metrics=baseline_metrics.baseline_str(),
        )

        success_status = (
            "SUCCESSFUL" if trained_metrics.is_training_successful() else "UNSUCCESSFUL"
        )
        logger.debug(
            "Trained model evaluation - Training",
            success_status=success_status,
            model_type="trained",
            model_path=self.model_path,
            training_status=success_status.lower(),
            quality_grade=trained_metrics.get_quality_grade(),
            metrics=str(trained_metrics),
        )

        return {"trained": trained_metrics, "baseline": baseline_metrics}

    def evaluate_with_summary(
        self, dataset_path: Path, max_samples: int | None = None
    ) -> dict[str, Any]:
        """Evaluate model and return comprehensive summary.

        Args:
            dataset_path: Path to evaluation dataset
            max_samples: Optional limit on samples

        Returns:
            Dictionary with metrics and human-readable summary
        """
        metrics = self.evaluate_dataset(dataset_path, max_samples, log_results=True)

        return {
            "metrics": metrics.to_dict(),
            "summary": {
                "model_path": self.model_path,
                "dataset_path": str(dataset_path),
                "samples_evaluated": metrics.num_samples,
                "training_successful": metrics.is_training_successful(),
                "quality_grade": metrics.get_quality_grade(),
                "positive_avg": f"{metrics.avg_positive_similarity:.4f}",
                "negative_avg": f"{metrics.avg_negative_similarity:.4f}",
                "ratio": f"{metrics.similarity_ratio:.4f}",
                "correlation": f"{metrics.spearman_correlation:.4f}",
            },
        }
