# Toxicity

Class for evaluating the toxicity of text through comprehensive multi-aspect analysis.

## Initialization

```python
class Toxicity:
    def __init__(
        self,
        summary: str,
        include_reason: bool = True,
        weights: Dict[str, float] = None,
    ):
```

## Hyperparameters

- **summary**: Text to be evaluated for toxicity
- **include_reason**: Whether to include detailed reasons in output (default: True)
- **weights**: Custom weights for different toxicity aspects
  - Default:
    - Hate Speech: 0.35
    - Profanity: 0.25
    - Personal Attacks: 0.25
    - Threat Level: 0.15

## Usage Example

```python
from toxicity import Toxicity
from languagemodels import LanguageModel

# Initialize the language model
llm = LanguageModel()

# Prepare text for toxicity analysis
text = "You're a terrible person who doesn't deserve any respect!"

# Create Toxicity instance
toxicity_metric = Toxicity(
    summary=text,
    include_reason=True,
    weights={
        "hate_speech": 0.4,
        "profanity": 0.2,
        "personal_attacks": 0.3,
        "threat_level": 0.1
    }
)

# Set the language model
toxicity_metric.set_model(llm)

# Perform toxicity evaluation
result = toxicity_metric.measure()

# Access the results
print(result['score'])  # Overall toxicity score
print(result['toxic_elements'])  # Identified toxic elements
print(result['toxicity_scores'])  # Detailed toxicity aspect scores
```

## Return Value

The `measure()` method returns a dictionary with:

- `score`: Overall toxicity score (0-1 range)
- `toxic_elements`: List of specific toxic elements found
- `toxicity_scores`: Detailed scores for different toxicity aspects
  - Each score contains:
    - `aspect`: Specific toxicity aspect
    - `score`: Numeric toxicity score
    - `reason`: Explanation of the score
    - `examples_found`: Optional list of specific examples
- `element_distribution`: Distribution of toxic elements across categories
- `verdict`: Detailed textual explanation of toxicity (if `include_reason` is True)

## Key Features

- Multi-aspect toxicity analysis
- Customizable evaluation weights
- Comprehensive toxicity assessment
- Flexible language model integration
- Detailed element identification and distribution
- Configurable reason inclusion

## Dependencies

- `typing`: For type hinting
- `pydantic`: For data validation
- `loguru`: For logging
- `json`: For response parsing

## Logging

The class logs token usage information, including:

- Total input tokens
- Total output tokens
- Total token count used during evaluation

## Error Handling

- Raises `ValueError` if:
  - No language model is set before measurement
  - Language model returns an empty response
- Provides robust JSON parsing
- Supports custom weight configuration

## Extensibility

The class is designed to be easily extended:

- Custom weight configuration
- Pluggable language model
- Flexible prompt templates
- Detailed logging and tracking

## Note

The Toxicity class provides a nuanced approach to content toxicity evaluation, focusing on multiple aspects of potentially harmful language while offering configurable analysis.
