# OutOfDistributionRobustness

Class for evaluating the out-of-distribution (OOD) robustness of language model outputs by analyzing how well the model handles inputs that differ significantly from its training distribution.

## Initialization

The `OutOfDistributionRobustness` class is initialized with the following parameters:

- **input_sentence**: The sentence to be evaluated for out-of-distribution robustness.

```python
class OutOfDistributionRobustness:
    def __init__(
        self,
        input_sentence: str,
    ):
        """
        Initialize the OutOfDistributionRobustness class to evaluate how well the model handles
        inputs that deviate from its typical training distribution.

        Parameters:
        input_sentence (str): The sentence to be evaluated for out-of-distribution robustness.
        """
        self.model = None
        self.template = OODRobustnessTemplate()
        self.input_sentence = input_sentence
        self.ood_robustness_score = 0
```

## Parameters Explanation

- **input_sentence**: The text input that needs to be evaluated for out-of-distribution robustness.

## Usage Example

Here is an example of how to use the `OutOfDistributionRobustness` class:

```python
from indoxJudge.metrics import OutOfDistributionRobustness
from indoxJudge.pipelines import Evaluator

# Define a sample input sentence
input_sentence = "The quantum fluctuations in the hyperdimensional matrix caused unexpected resonance."

# Initialize the OutOfDistributionRobustness object
ood_robustness = OutOfDistributionRobustness(
    input_sentence=input_sentence
)

# Set up the evaluator
evaluator = Evaluator(model=language_model, metrics=[ood_robustness])

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

- The OOD robustness evaluation assesses how well the model handles unusual, rare, or novel inputs.
- The evaluation process tests the model's ability to maintain reliable performance on inputs that differ from typical training examples.
- The class uses a default OODRobustnessTemplate for evaluation criteria and prompts.
