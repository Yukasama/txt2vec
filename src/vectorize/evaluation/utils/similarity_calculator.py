"""Similarity computation utilities for evaluation."""

import numpy as np
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

__all__ = ["SimilarityCalculator"]


class SimilarityCalculator:
    """Handles similarity computations for evaluation."""

    @staticmethod
    def compute_cosine_similarities(
        model: SentenceTransformer,
        questions: list[str],
        positives: list[str],
        negatives: list[str],
    ) -> tuple[list[float], list[float]]:
        """Compute cosine similarities for question-positive-negative triplets.

        Efficiently computes pairwise cosine similarities using vectorized operations
        for better performance on large datasets.

        Args:
            model: SentenceTransformer model
            questions: List of question texts
            positives: List of positive answer texts
            negatives: List of negative answer texts

        Returns:
            Tuple of (positive_similarities, negative_similarities)

        Raises:
            ValueError: If input lists have different lengths
        """
        if not (len(questions) == len(positives) == len(negatives)):
            raise ValueError(
                f"Input lists must have same length: "
                f"questions={len(questions)}, positives={len(positives)}, "
                f"negatives={len(negatives)}"
            )

        question_embeddings = model.encode(questions, show_progress_bar=False)
        positive_embeddings = model.encode(positives, show_progress_bar=False)
        negative_embeddings = model.encode(negatives, show_progress_bar=False)

        positive_similarities = np.diag(
            cosine_similarity(question_embeddings, positive_embeddings)
        ).tolist()

        negative_similarities = np.diag(
            cosine_similarity(question_embeddings, negative_embeddings)
        ).tolist()

        return positive_similarities, negative_similarities

    @staticmethod
    def compute_spearman_correlation(
        positive_similarities: list[float],
        negative_similarities: list[float],
    ) -> float:
        """Compute Spearman correlation for similarity ranking.

        Evaluates how well the model ranks positive examples higher than
        negative ones. Higher correlation indicates better discrimination
        between positive and negative examples.

        Args:
            positive_similarities: List of positive similarities
            negative_similarities: List of negative similarities

        Returns:
            Spearman correlation coefficient (0.0 if correlation cannot be computed)

        Raises:
            ValueError: If input lists are empty
        """
        if not positive_similarities or not negative_similarities:
            raise ValueError("Similarity lists cannot be empty")

        expected_scores = [1] * len(positive_similarities) + [0] * len(
            negative_similarities
        )
        actual_scores = positive_similarities + negative_similarities

        if len(set(actual_scores)) <= 1:
            return 0.0

        try:
            correlation_result = spearmanr(expected_scores, actual_scores)
            correlation_value = correlation_result[0]  # type: ignore[index]

            if correlation_value is None:
                return 0.0

            correlation_float = float(correlation_value)  # type: ignore[arg-type]
            return correlation_float if not np.isnan(correlation_float) else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def compute_similarity_ratio(
        avg_positive: float,
        avg_negative: float,
    ) -> float:
        """Compute similarity ratio (positive/negative).

        Args:
            avg_positive: Average positive similarity
            avg_negative: Average negative similarity

        Returns:
            Similarity ratio or infinity if negative is zero
        """
        return avg_positive / avg_negative if avg_negative > 0 else float("inf")
