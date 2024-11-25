# StereotypeBias

Class for evaluating the presence of stereotypical biases in language model outputs by analyzing content for social, cultural, gender, racial, and other forms of stereotyping.

## Initialization

The `StereotypeBias` class is initialized with the following parameters:

- **input_sentence**: The sentence to be evaluated for stereotypical biases.

```python
class StereotypeBias:
    def __init__(
        self,
        input_sentence: str,
    ):
        """
        Initialize the StereotypeBias class to evaluate the presence of stereotypical biases
        in language model outputs, including social, cultural, gender, and racial stereotypes.

        Parameters:
        input_sentence (str): The sentence to be evaluated for stereotypical biases.
        """
        self.model = None
        self.template = StereotypeBiasTemplate()
        self.input_sentence = input_sentence
        self.stereotype_bias_score = 0
```

## Parameters Explanation

- **input_sentence**: The text input that needs to be evaluated for stereotypical bias content.

## Usage Example

Here is an example of how to use the `StereotypeBias` class:

```python
from indoxJudge.metrics import StereotypeBias
from indoxJudge.pipelines import Evaluator

# Define a sample input sentence
input_sentence = "The engineering team consists of technical guys while the HR department is managed by ladies."

# Initialize the StereotypeBias object
bias = StereotypeBias(
    input_sentence=input_sentence
)

# Set up the evaluator
evaluator = Evaluator(model=language_model, metrics=[bias])

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

- The stereotype bias evaluation examines various aspects including gender stereotypes, racial stereotypes, age-related stereotypes, and cultural stereotypes.
- The evaluation process identifies potential biased assumptions and generalizations in model outputs.
- The class uses a default StereotypeBiasTemplate for evaluation criteria and prompts.
