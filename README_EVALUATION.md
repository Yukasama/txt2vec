# Evaluation Module

This module provides comprehensive evaluation capabilities for SBERT (Sentence-BERT) training quality assessment using cosine similarity metrics.

## Overview

The evaluation system computes various metrics to assess how well a trained sentence transformer model performs on question-positive-negative triplets:

- **Average Cosine Similarity**: Between questions and positive/negative examples
- **Similarity Ratio**: Ratio of positive to negative similarities (should be > 1.2 for good training)
- **Spearman Correlation**: Measures ranking quality between positive and negative examples
- **Quality Grading**: Automatic quality assessment (Excellent, Good, Fair, Poor)

## Architecture

### Core Components

1. **`evaluation.py`**: Main evaluation logic
   - `EvaluationMetrics`: Container for computed metrics
   - `TrainingEvaluator`: Main evaluation orchestrator

2. **`service.py`**: Service layer for FastAPI integration
   - Database lookups for models and datasets
   - Error handling and validation

3. **`router.py`**: FastAPI endpoints
   - RESTful evaluation API

4. **`schemas.py`**: Pydantic models for API requests/responses

### Utility Modules (`utils/`)

1. **`similarity_calculator.py`**: Optimized similarity computation
   - Vectorized cosine similarity calculations
   - Spearman correlation computation
   - Performance optimized for large datasets

2. **`dataset_validator.py`**: Dataset validation
   - JSONL format validation
   - Required columns checking
   - Null/empty value detection

3. **`model_resolver.py`**: Model path resolution
   - Handles HuggingFace cache structure
   - Recursive model file discovery

## Usage Examples

### Basic Evaluation

```python
from vectorize.evaluation import TrainingEvaluator
from pathlib import Path

# Initialize evaluator
evaluator = TrainingEvaluator("data/models/my-model")

# Evaluate on dataset
metrics = evaluator.evaluate_dataset(
    dataset_path=Path("data/datasets/eval.jsonl"),
    max_samples=1000
)

print(f"Training successful: {metrics.is_training_successful()}")
print(f"Quality grade: {metrics.get_quality_grade()}")
print(f"Similarity ratio: {metrics.similarity_ratio:.3f}")
```

### Model Comparison

```python
# Compare trained model against baseline
comparison = evaluator.compare_models(
    baseline_model_path="sentence-transformers/all-MiniLM-L6-v2",
    dataset_path=Path("data/datasets/eval.jsonl")
)

trained_metrics = comparison["trained"]
baseline_metrics = comparison["baseline"]

# Get improvement metrics
improvement = trained_metrics.get_improvement_over_baseline(baseline_metrics)
print(f"Ratio improvement: {improvement['ratio_improvement']:.3f}")
```

### API Usage

```bash
# Evaluate model via REST API
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_tag": "my-trained-model",
    "dataset_id": "uuid-of-dataset",
    "max_samples": 1000,
    "baseline_model_tag": "baseline-model"
  }'
```

## Metrics Interpretation

### Quality Grades

- **Excellent**: Ratio ≥ 2.0 and Correlation ≥ 0.7
- **Good**: Ratio ≥ 1.5 and Correlation ≥ 0.5  
- **Fair**: Ratio ≥ 1.2 and Correlation ≥ 0.3
- **Poor**: Below Fair thresholds

### Training Success Criteria

Training is considered successful if:
1. Positive similarities > negative similarities
2. Similarity ratio > 1.2

### Similarity Ratio

- **> 2.0**: Excellent discrimination
- **1.5-2.0**: Good discrimination
- **1.2-1.5**: Acceptable discrimination
- **< 1.2**: Poor discrimination (training likely unsuccessful)

### Spearman Correlation

- **> 0.7**: Excellent ranking quality
- **0.5-0.7**: Good ranking quality
- **0.3-0.5**: Fair ranking quality
- **< 0.3**: Poor ranking quality

## Configuration

Key constants in `evaluation.py`:

```python
TRAINING_SUCCESS_THRESHOLD = 1.2  # Minimum ratio for success
DEFAULT_MAX_SAMPLES = 1000        # Default sample limit
DEFAULT_RANDOM_SEED = 42          # For reproducible sampling
```

## Error Handling

The module provides comprehensive error handling:

- `DatasetValidationError`: Invalid or malformed datasets
- `ModelNotFoundError`: Missing model files
- `InvalidDatasetIdError`: Invalid dataset UUIDs
- `TrainingDatasetNotFoundError`: Dataset files not found

## Performance Considerations

### Optimizations

1. **Vectorized Similarity Computation**: Uses numpy operations instead of loops
2. **Batch Encoding**: Encodes all texts at once for efficiency
3. **Memory Management**: Handles large datasets efficiently
4. **Sample Limiting**: Supports limiting evaluation samples for speed

### Recommended Limits

- **Small models**: Up to 10,000 samples
- **Large models**: 1,000-5,000 samples for reasonable performance
- **Production**: Consider async evaluation for large datasets

## Testing

Run evaluation tests:

```bash
pytest tests/evaluation/ -v
```

Test categories:
- **Unit tests**: Individual component testing
- **Integration tests**: End-to-end evaluation (requires models)
- **Validation tests**: Dataset validation edge cases

## Dependencies

- **sentence-transformers**: Model loading and inference
- **scikit-learn**: Cosine similarity computation
- **scipy**: Spearman correlation
- **pandas**: Dataset manipulation
- **numpy**: Numerical operations
- **loguru**: Structured logging
