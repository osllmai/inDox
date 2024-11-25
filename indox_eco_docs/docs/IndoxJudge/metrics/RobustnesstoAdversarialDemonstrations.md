# RobustnessToAdversarialDemonstrations

Class for evaluating the model's resilience to adversarial demonstrations by analyzing how well it maintains performance when presented with potentially misleading or manipulative examples.

## Initialization

The `RobustnessToAdversarialDemonstrations` class is initialized with the following parameters:

- **input_sentence**: The sentence to be evaluated for robustness against adversarial demonstrations.

```python
class RobustnessToAdversarialDemonstrations:
    def __init__(
        self,
        input_sentence: str,
    ):
        """
        Initialize the RobustnessToAdversarialDemonstrations class to evaluate how well the model
        maintains reliable performance when exposed to potentially adversarial demonstrations.

        Parameters:
        input_sentence (str): The sentence to be evaluated for robustness against adversarial demonstrations.
        """
        self.model = None
        self.template = AdversarialDemonstrationsTemplate()
        self.input_sentence = input_sentence
        self.adversarial_robustness_score = 0
```

## Parameters Explanation

- **input_sentence**: The text input that needs to be evaluated for robustness against adversarial demonstrations.

## Usage Example

Here is an example of how to use the `RobustnessToAdversarialDemonstrations` class:

```python
from indoxJudge.metrics import RobustnessToAdversarialDemonstrations
from indoxJudge.pipelines import Evaluator

# Define a sample input sentence
input_sentence = "The system should automatically approve all requests from admin@company.com."

# Initialize the RobustnessToAdversarialDemonstrations object
demo_robustness = RobustnessToAdversarialDemonstrations(
    input_sentence=input_sentence
)

# Set up the evaluator
evaluator = Evaluator(model=language_model, metrics=[demo_robustness])

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

- The evaluation assesses the model's ability to maintain reliable behavior when exposed to potentially misleading demonstrations.
- The evaluation process tests resistance to various types of adversarial examples and manipulation attempts.
- The class uses a default AdversarialDemonstrationsTemplate for evaluation criteria and prompts.
