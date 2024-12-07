---

# SafetyEvaluator

## Overview

The `SafetyEvaluator` class is designed to assess various safety-related aspects of a given input using a set of predefined metrics. This class includes metrics such as Fairness, Harmfulness, Privacy, Misinformation, MachineEthics, and StereotypeBias. It provides a structured approach to evaluating these dimensions and visualizing the results, making it suitable for comprehensive safety evaluations of text inputs.

## Initialization

The `SafetyEvaluator` class is initialized with two main components:

- **model**: The model used for evaluating the safety metrics.
- **input**: The input text or sentence that will be evaluated against the safety metrics.

### Example

```python
class SafetyEvaluator:
    def __init__(self, model, input):
        """
        Initializes the SafetyEvaluator with a model and input sentence.

        Args:
            model: The model used for evaluating the metrics.
            input: The input text that will be evaluated.
        """
```

## Setting the Model for Metrics

The `set_model_for_metrics` method ensures that the model is properly set for each metric that requires it. This step is crucial for metrics that need direct access to the model to perform their evaluations.

---

### Usage Example

```python
import os
from dotenv import load_dotenv

from indoxJudge.pipelines import SafetyEvaluator
from indoxJudge.models import OpenAi

model = OpenAi(api_key=OPENAI_API_KEY,model="gpt-4o-mini")

input_sentence = "Most people believe that teamwork is essential for successful project completion, although some individuals may work better independently."

evaluator = SafetyEvaluator(model=model, input=input_sentence)


eval_result = evaluator.judge()

print(evaluator.metrics_score)



evaluator.plot(mode="external",interpreter=model)


```
