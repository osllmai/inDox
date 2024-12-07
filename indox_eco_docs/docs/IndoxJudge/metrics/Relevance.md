# Relevance

Class for evaluating the relevance of a summary against its source text through comprehensive analysis.

## Initialization

```python
class Relevance:
    def __init__(
        self,
        summary: str,
        source_text: str,
        include_reason: bool = True,
        weights: Dict[str, float] = None,
    ):
```

## Hyperparameters

- **summary**: Summary text being evaluated
- **source_text**: Original text for comparison
- **include_reason**: Whether to include detailed reasons in output (default: True)
- **weights**: Custom weights for different relevance aspects
  - Default:
    - Key Information Coverage: 0.4
    - Topic Alignment: 0.3
    - Information Accuracy: 0.2
    - Focus Distribution: 0.1

## Usage Example

```python
from relevance import Relevance
from languagemodels import LanguageModel

# Initialize the language model
llm = LanguageModel()

# Prepare source text and summary
source_text = "Paris is the capital of France, known for its rich history, art, and culture."
summary = "Paris, a global city in France, is renowned for its artistic heritage and cultural significance."

# Create Relevance instance
relevance_metric = Relevance(
    summary=summary,
    source_text=source_text,
    include_reason=True,
    weights={
        "key_information_coverage": 0.45,
        "topic_alignment": 0.35,
        "information_accuracy": 0.15,
        "focus_distribution": 0.05
    }
)

# Set the language model
relevance_metric.set_model(llm)

# Perform relevance evaluation
result = relevance_metric.measure()

# Access the results
print(result['score'])  # Overall relevance score
print(result['key_points'])  # Extracted key points
print(result['key_point_coverage'])  # Coverage of key points
```

## Return Value

The `measure()` method returns a dictionary with:

- `score`: Overall relevance score
- `key_points`: Extracted key points from source text
- `relevance_scores`: Detailed scores for different relevance aspects
- `key_point_coverage`: Mapping of key points coverage

## Key Features

- Granular key point tracking
- Aspect-based relevance analysis
- Importance-weighted scoring
- Comprehensive relevance evaluation
- Detailed key point coverage analysis
