# Fairness

Class for evaluating the fairness of language model outputs by analyzing potential biases, stereotypes, and discriminatory content using a specified language model.

## Initialization

The `Fairness` class is initialized with the following parameters:

- **input_sentence**: The sentence to be evaluated for fairness.

```python
class Fairness:
    def __init__(
        self,
        input_sentence: str,
    ):
        """
        Initialize the Fairness class to evaluate potential biases, stereotypes, and discriminatory content
        in language model outputs.

        Parameters:
        input_sentence (str): The sentence to be evaluated for fairness.
        """
        self.model = None
        self.template = FairnessTemplate()
        self.input_sentence = input_sentence
        self.fairness_score = 0
```

## Parameters Explanation

- **input_sentence**: The text input that needs to be evaluated for fairness concerns.

## Usage Example

Here is an example of how to use the `Fairness` class:

```python
from indoxJudge.metrics import Fairness
from indoxJudge.pipelines import Evaluator

# Define a sample input sentence
input_sentence = "The company is looking for energetic young professionals."

# Initialize the Fairness object
fairness = Fairness(
    input_sentence=input_sentence
)

# Set up the evaluator
evaluator = Evaluator(model=language_model, metrics=[fairness])

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

- The fairness evaluation examines various aspects including gender bias, age discrimination, racial bias, and other forms of prejudice.
- The evaluation process uses the specified language model to identify potential fairness concerns.
- The class uses a default FairnessTemplate for evaluation criteria and prompts.
