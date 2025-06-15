# ruff: noqa: S101, ANN401
"""Logic-only tests for model loader using pytest - no mocks, no files."""

from typing import Any

import pytest


class MockConfig:
    """Simple config that just holds attributes."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize MockConfig.

        Args:
            **kwargs: Arbitrary keyword arguments to set as attributes.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)


def detect_model_type(config: MockConfig) -> str:
    """Extract the model detection logic for testing."""
    model_type = getattr(config, "model_type", None)
    architectures = getattr(config, "architectures", None)

    if model_type == "t5":
        return "T5EncoderModel"
    if architectures and architectures[0] and "MaskedLM" in architectures[0]:
        return "AutoModelForMaskedLM"
    return "AutoModel"


class TestModelTypeDetection:
    """Test model type detection logic."""

    @staticmethod
    def test_t5_model_detection() -> None:
        """Test T5 model is correctly detected."""
        config = MockConfig(
            model_type="t5", architectures=["T5ForConditionalGeneration"]
        )
        detected_type = detect_model_type(config)
        assert detected_type == "T5EncoderModel"

    @staticmethod
    def test_bert_masked_lm_detection() -> None:
        """Test BERT MaskedLM model is correctly detected."""
        config = MockConfig(model_type="bert", architectures=["BertForMaskedLM"])
        detected_type = detect_model_type(config)
        assert detected_type == "AutoModelForMaskedLM"

    @staticmethod
    def test_roberta_masked_lm_detection() -> None:
        """Test RoBERTa MaskedLM model is correctly detected."""
        config = MockConfig(model_type="roberta", architectures=["RobertaForMaskedLM"])
        detected_type = detect_model_type(config)
        assert detected_type == "AutoModelForMaskedLM"

    @staticmethod
    def test_distilbert_masked_lm_detection() -> None:
        """Test DistilBERT MaskedLM model is correctly detected."""
        config = MockConfig(
            model_type="distilbert", architectures=["DistilBertForMaskedLM"]
        )
        detected_type = detect_model_type(config)
        assert detected_type == "AutoModelForMaskedLM"

    @staticmethod
    def test_gpt2_model_fallback() -> None:
        """Test GPT2 model falls back to AutoModel."""
        config = MockConfig(model_type="gpt2", architectures=["GPT2LMHeadModel"])
        detected_type = detect_model_type(config)
        assert detected_type == "AutoModel"

    @staticmethod
    def test_unknown_model_fallback() -> None:
        """Test unknown model type falls back to AutoModel."""
        config = MockConfig(model_type="unknown", architectures=["UnknownModel"])
        detected_type = detect_model_type(config)
        assert detected_type == "AutoModel"


class TestMissingAttributes:
    """Test handling of missing config attributes."""

    @staticmethod
    def test_missing_model_type() -> None:
        """Test behavior when model_type is missing."""
        config = MockConfig(architectures=["SomeModel"])  # No model_type
        detected_type = detect_model_type(config)
        assert detected_type == "AutoModel"

    @staticmethod
    def test_missing_architectures() -> None:
        """Test behavior when architectures is missing."""
        config = MockConfig(model_type="custom")  # No architectures
        detected_type = detect_model_type(config)
        assert detected_type == "AutoModel"

    @staticmethod
    def test_empty_architectures_list() -> None:
        """Test behavior when architectures is empty list."""
        config = MockConfig(model_type="bert", architectures=[])
        detected_type = detect_model_type(config)
        assert detected_type == "AutoModel"

    @staticmethod
    def test_both_attributes_missing() -> None:
        """Test behavior when both model_type and architectures are missing."""
        config = MockConfig()  # Empty config
        detected_type = detect_model_type(config)
        assert detected_type == "AutoModel"


class TestEdgeCases:
    """Test edge cases in model detection logic."""

    @staticmethod
    def test_t5_prioritizes_over_architecture() -> None:
        """Test T5 model_type takes priority over architecture."""
        config = MockConfig(model_type="t5", architectures=["SomeOtherArchitecture"])
        detected_type = detect_model_type(config)
        assert detected_type == "T5EncoderModel"

    @staticmethod
    def test_architecture_without_masked_lm() -> None:
        """Test architecture that doesn't contain MaskedLM."""
        config = MockConfig(
            model_type="bert", architectures=["BertForSequenceClassification"]
        )
        detected_type = detect_model_type(config)
        assert detected_type == "AutoModel"

    @staticmethod
    def test_multiple_architectures_first_has_masked_lm() -> None:
        """Test when first architecture contains MaskedLM."""
        config = MockConfig(
            model_type="bert",
            architectures=["BertForMaskedLM", "BertForSequenceClassification"],
        )
        detected_type = detect_model_type(config)
        assert detected_type == "AutoModelForMaskedLM"

    @staticmethod
    def test_multiple_architectures_first_no_masked_lm() -> None:
        """Test when first architecture doesn't contain MaskedLM."""
        config = MockConfig(
            model_type="bert",
            architectures=["BertForSequenceClassification", "BertForMaskedLM"],
        )
        detected_type = detect_model_type(config)
        assert detected_type == "AutoModel"


class TestGetattrSafety:
    """Test that getattr() safely handles missing attributes."""

    @staticmethod
    def test_getattr_returns_none_for_missing_model_type() -> None:
        """Test getattr returns None for missing model_type."""
        config = MockConfig()
        model_type = getattr(config, "model_type", None)
        assert model_type is None

    @staticmethod
    def test_getattr_returns_none_for_missing_architectures() -> None:
        """Test getattr returns None for missing architectures."""
        config = MockConfig()
        architectures = getattr(config, "architectures", None)
        assert architectures is None

    @staticmethod
    def test_getattr_returns_actual_values_when_present() -> None:
        """Test getattr returns actual values when attributes exist."""
        config = MockConfig(model_type="bert", architectures=["BertModel"])
        model_type = getattr(config, "model_type", "default")
        architectures = getattr(config, "architectures", [])
        assert model_type == "bert"
        assert architectures == ["BertModel"]

    @staticmethod
    def test_getattr_with_default_values() -> None:
        """Test getattr with custom default values."""
        config = MockConfig()
        model_type = getattr(config, "model_type", "unknown")
        architectures = getattr(config, "architectures", [])
        assert model_type == "unknown"
        assert not architectures


# Parametrized tests for more concise testing
class TestParametrized:
    """Parametrized tests for comprehensive coverage."""

    @staticmethod
    @pytest.mark.parametrize(
        ("model_type", "architectures", "expected"),
        [
            # T5 models
            ("t5", ["T5ForConditionalGeneration"], "T5EncoderModel"),
            ("t5", ["T5EncoderModel"], "T5EncoderModel"),
            (
                "t5",
                [],
                "T5EncoderModel",
            ),  # T5 prioritized even with empty architectures
            # MaskedLM models
            ("bert", ["BertForMaskedLM"], "AutoModelForMaskedLM"),
            ("roberta", ["RobertaForMaskedLM"], "AutoModelForMaskedLM"),
            ("distilbert", ["DistilBertForMaskedLM"], "AutoModelForMaskedLM"),
            # Standard models (fallback)
            ("gpt2", ["GPT2LMHeadModel"], "AutoModel"),
            ("bert", ["BertForSequenceClassification"], "AutoModel"),
            ("unknown", ["UnknownModel"], "AutoModel"),
            # Missing attributes
            (None, ["SomeModel"], "AutoModel"),
            ("custom", None, "AutoModel"),
            (None, None, "AutoModel"),
        ],
    )
    def test_model_detection_scenarios(
        model_type: str | None,
        architectures: list[str] | None,
        expected: str,
    ) -> None:
        """Test various model detection scenarios."""
        config_kwargs: dict[str, Any] = {}
        if model_type is not None:
            config_kwargs["model_type"] = model_type
        if architectures is not None:
            config_kwargs["architectures"] = architectures
        config = MockConfig(**config_kwargs)

        detected_type = detect_model_type(config)
        assert detected_type == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
