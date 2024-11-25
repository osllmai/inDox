# Privacy

Class for evaluating the privacy considerations of language model outputs by analyzing potential personal information disclosure, sensitive data handling, and privacy compliance.

## Initialization

The `Privacy` class is initialized with the following parameters:

- **input_sentence**: The sentence to be evaluated for privacy concerns.

```python
class Privacy:
    def __init__(
        self,
        input_sentence: str,
    ):
        """
        Initialize the Privacy class to evaluate potential privacy concerns, sensitive information
        disclosure, and data protection compliance in language model outputs.

        Parameters:
        input_sentence (str): The sentence to be evaluated for privacy considerations.
        """
        self.model = None
        self.template = PrivacyTemplate()
        self.input_sentence = input_sentence
        self.privacy_score = 0
```

## Parameters Explanation

- **input_sentence**: The text input that needs to be evaluated for privacy concerns.

## Usage Example

Here is an example of how to use the `Privacy` class:

```python
from indoxJudge.metrics import Privacy
from indoxJudge.pipelines import Evaluator

# Define a sample input sentence
input_sentence = "John Smith's account number is 1234-5678-9012, and his email is john@example.com."

# Initialize the Privacy object
privacy = Privacy(
    input_sentence=input_sentence
)

# Set up the evaluator
evaluator = Evaluator(model=language_model, metrics=[privacy])

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

- The privacy evaluation examines various aspects including personal identifiable information (PII), sensitive data exposure, and compliance with privacy standards.
- The evaluation process identifies potential privacy risks in model outputs.
- The class uses a default PrivacyTemplate for evaluation criteria and prompts.
