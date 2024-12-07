# ROUGE Score

Class for evaluating the quality of generated summaries by comparing them to reference summaries using the ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric.

## Initialization

```python
class Rouge:
    def __init__(
        self,
        generated_summary: str,
        reference_summary: str,
        include_reason: bool = True,
        weights: Dict[str, float] = None,
        skip_window: int = 4,
    ):
```

## Hyperparameters

- **generated_summary**: Generated text to be evaluated
- **reference_summary**: Original reference text for comparison
- **include_reason**: Whether to include detailed reasons in output (default: True)
- **weights**: Weights for different ROUGE metrics (default: `{"rouge_1": 0.6, "rouge_2": 0.25, "rouge_l": 0.1, "rouge_s": 0.05}`)
- **skip_window**: Maximum gap size for skip-bigrams (default: 4)

## Usage Example

```python
from rouge import Rouge
from languagemodels import LanguageModel

# Initialize the language model
llm = LanguageModel()

# Prepare generated and reference summaries
generated_summary = "The quick brown fox jumps over the lazy dog."
reference_summary = "A fast brown fox leaps above a sleepy canine."

# Create Rouge instance
rouge_metric = Rouge(
    generated_summary=generated_summary,
    reference_summary=reference_summary,
    include_reason=True
)

# Set the language model
rouge_metric.set_model(llm)

# Perform ROUGE score evaluation
result = rouge_metric.measure()

# Access the results
print(result['overall_score'])  # Overall ROUGE score
print(result['detailed_scores'])  # Detailed ROUGE metrics
print(result['verdict'])  # Textual explanation of summary quality
```

## Return Value

The `measure()` method returns a dictionary with:

- `overall_score`: Weighted average ROUGE score
- `detailed_scores`: Comprehensive ROUGE metric details
  - **ROUGE-1**: Unigram overlap assessment
  - **ROUGE-2**: Bigram overlap assessment
  - **ROUGE-L**: Longest common subsequence assessment
  - **ROUGE-S**: Skip-bigram overlap assessment
- `verdict`: Detailed textual explanation of summary quality (if `include_reason` is True)

Each metric provides:

- `precision`: Precision score
- `recall`: Recall score
- `f1_score`: F1 score
- `details`: Additional evaluation metrics specific to each ROUGE variant

## ROUGE Variants

1. **ROUGE-1**: Unigram overlap

   - Measures the overlap of individual words
   - Simple yet effective for basic similarity

2. **ROUGE-2**: Bigram overlap

   - Captures two-word phrase matches
   - More context-aware than unigrams

3. **ROUGE-L**: Longest Common Subsequence

   - Finds the longest co-occurring in-sequence n-grams
   - Accounts for sentence structure similarity

4. **ROUGE-S**: Skip-bigram
   - Allows non-consecutive word pair matches
   - Flexible matching with configurable window size

## Preprocessing Techniques

- Lowercase conversion
- Special character removal
- Tokenization
- Sentence-level processing
- N-gram extraction
- Skip-bigram calculation

## Scoring Methodology

- Uses modified precision and recall calculation
- Applies smoothing techniques
- Incorporates length penalty
- Supports weighted scoring across different ROUGE metrics

## Computational Details

- Calculates n-gram and skip-bigram overlaps
- Implements longest common subsequence algorithm
- Provides detailed computational metrics

## Logging

The class provides token usage tracking:

- Total input tokens
- Total output tokens
- Detailed token count information

## Error Handling

- Raises `ValueError` if:
  - Language model returns an empty response
- Robust preprocessing and scoring mechanism
- Graceful handling of text variations

## Extensibility

The class is designed to be easily extended:

- Customizable ROUGE metric weights
- Flexible preprocessing
- Pluggable language model for verdicts

## Dependencies

- `nltk`: Natural Language Toolkit
- `pydantic`: Data validation
- `loguru`: Logging
- `json`: Response parsing
- `numpy`: Numerical computations
- `re`: Regular expression processing

## Note

ROUGE score provides a comprehensive method for evaluating summary quality, particularly useful in:

- Text summarization
- Machine translation evaluation
- Content generation assessment

The metric goes beyond simple word matching by considering various levels of textual similarity and structural coherence.
