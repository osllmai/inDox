# FactualConsistency

Class for evaluating the factual consistency between a summary and its source text.

## Initialization

```python
class FactualConsistency:
    def __init__(
        self,
        summary: str,
        source_text: str,
        category_weights: Dict[str, float] = None,
        consistency_threshold: float = 0.8,
    ):
```

## Hyperparameters

- **summary**: Text to be evaluated for factual consistency
- **source_text**: Original source for comparison
- **category_weights**: Custom weights for different claim categories
  - Default:
    - Numerical claims: 0.25
    - Entity claims: 0.25
    - Causal claims: 0.20
    - Descriptive claims: 0.15
    - Comparative claims: 0.15
- **consistency_threshold**: Minimum score for considering a claim consistent (default: 0.8)

## Key Metrics

1. **Claim Extraction**

   - Identifies individual claims in the summary
   - Verifies each claim against source text

2. **Consistency Scoring**
   - Assigns scores to claims
   - Calculates category-level verdicts
   - Generates overall consistency score

## Usage Example

```python
from indoxJudge.metrics import FactualConsistency
from indoxJudge.models import YourLanguageModel

llm = YourLanguageModel()
summary = "Paris is the capital of France with 2.1 million residents."
source_text = "Paris, located in France, is the national capital with a population of approximately 2.16 million."

factual_metric = FactualConsistency(
    summary=summary,
    source_text=source_text,
    consistency_threshold=0.8
)
factual_metric.set_model(llm)
result = factual_metric.measure()
```

## Return Value

The `measure()` method returns a dictionary with:

- `score`: Overall factual consistency score
- `summary_claims`: Extracted summary claims
- `verified_claims`: Detailed claim verification
- `category_scores`: Scores by claim category
- `consistency_stats`: Comprehensive consistency statistics

## Key Features

- Detailed claim-level analysis
- Customizable category weighting
- Error type identification
- Comprehensive statistical reporting
