# ToxicityDiscriminative

Class for evaluating and discriminating toxic content in one or multiple text inputs by analyzing harmful, offensive, or inappropriate content using configurable thresholds and modes.

## Initialization

The `ToxicityDiscriminative` class is initialized with the following parameters:

- **texts**: Single text string or list of text strings to be evaluated for toxicity.
- **threshold**: Threshold value for toxicity classification (default: 0.5).
- **include_reason**: Flag to include detailed reasoning in results (default: True).
- **strict_mode**: Enable strict toxicity evaluation mode (default: False).

```python
class ToxicityDiscriminative:
    def __init__(
        self,
        texts: Union[str, List[str]],
        threshold: float = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False
    ):
        """
        Initialize the ToxicityDiscriminative class to evaluate and discriminate toxic content
        in one or multiple text inputs with configurable evaluation parameters.

        Parameters:
        texts (Union[str, List[str]]): Single text string or list of text strings to evaluate.
        threshold (float): Threshold value for toxicity classification. Defaults to 0.5.
                          Set to 0 if strict_mode is True.
        include_reason (bool): Whether to include detailed reasoning in results. Defaults to True.
        strict_mode (bool): Enable strict evaluation mode. Defaults to False.
        """
        self.model = None
        self.threshold = 0 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.texts = [texts] if isinstance(texts, str) else texts
```

## Parameters Explanation

- **texts**: Input text or list of texts to be evaluated for toxicity content.
- **threshold**: Numerical threshold for determining toxic content (ignored if strict_mode is True).
- **include_reason**: When True, provides detailed explanations for toxicity classifications.
- **strict_mode**: When True, enforces stricter toxicity evaluation with zero threshold.

## Usage Example

Here is an example of how to use the `ToxicityDiscriminative` class:

```python
from indoxJudge.metrics import ToxicityDiscriminative
from indoxJudge.pipelines import Evaluator

# Define sample texts for evaluation
texts = [
    "This is a normal review of the product.",
    "This product is absolutely terrible!",
    "The customer service was very helpful."
]

# Initialize the ToxicityDiscriminative object
toxicity_discriminator = ToxicityDiscriminative(
    texts=texts,
    threshold=0.7,
    include_reason=True,
    strict_mode=False
)

# Set up the evaluator
evaluator = Evaluator(model=language_model, metrics=[toxicity_discriminator])

# Get the evaluation results
results = evaluator.judge()
```

## Error Handling

The class implements comprehensive error handling for:

- Invalid model responses
- JSON parsing errors
- Invalid input formats
- Threshold validation
- Input text validation

## Notes

- The discriminative evaluation provides separate toxicity assessments for each input text.
- Strict mode enforces a zero threshold for maximum sensitivity to toxic content.
- The class provides both individual and aggregate toxicity analysis.
- When include_reason is True, detailed explanations are provided for each classification.
