# SafetyToxicity

Class for evaluating the toxicity level of language model outputs by analyzing harmful, offensive, or inappropriate content using a specified language model.

## Initialization

The `SafetyToxicity` class is initialized with the following parameters:

- **input_sentence**: The sentence to be evaluated for toxicity and safety concerns.

```python
class SafetyToxicity:
    def __init__(
        self,
        input_sentence: str,
    ):
        """
        Initialize the SafetyToxicity class to evaluate the level of harmful, offensive,
        or inappropriate content in language model outputs.

        Parameters:
        input_sentence (str): The sentence to be evaluated for toxicity and safety concerns.
        """
        self.model = None
        self.template = ToxicityTemplate()
        self.input_sentence = input_sentence
        self.toxicity_score = 0
```

## Parameters Explanation

- **input_sentence**: The text input that needs to be evaluated for toxicity and safety issues.

## Usage Example

Here is an example of how to use the `SafetyToxicity` class:

```python
from indoxJudge.metrics import SafetyToxicity
from indoxJudge.pipelines import Evaluator

# Define a sample input sentence
input_sentence = "The product review contains strong language about customer service."

# Initialize the SafetyToxicity object
toxicity = SafetyToxicity(
    input_sentence=input_sentence
)

# Set up the evaluator
evaluator = Evaluator(model=language_model, metrics=[toxicity])

# Get the evaluation results
results = evaluator.judge()
```

## Error Handling

The class implements comprehensive error handling for:

- Invalid model responses
- JSON parsing errors
- Template rendering issues
- Invalid input formats

## Notes

- The toxicity evaluation examines various aspects including hate speech, profanity, harmful content, and inappropriate language.
- The evaluation process identifies potential safety concerns in model outputs.
- The class uses a default ToxicityTemplate for evaluation criteria and prompts.
