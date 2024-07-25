# Evaluator

## Overview

The `Evaluator` class is designed to evaluate various aspects of language model outputs using a range of metrics. It
supports metrics such as Faithfulness, Answer Relevancy, Bias, Contextual Relevancy, GEval, Hallucination, Knowledge
Retention, Toxicity, BertScore, BLEU, Rouge, and METEOR. This class is ideal for assessing different dimensions of model
performance and ensuring comprehensive evaluation.

## Initialization

The `Evaluator` class is initialized with two main components:

- **Model**: The language model to be evaluated.
- **Metrics**: A list of metric instances used to evaluate the model.

### Example

```python
from typing import List


class Evaluator:
    def __init__(self, model, metrics: List):
        """
        Initializes the Evaluator with a language model and a list of metrics.

        Args:
            model: The language model to be evaluated.
            metrics (List): A list of metric instances to evaluate the model.
        """
```

## Setting the Model for Metrics

The `set_model_for_metrics` method ensures that the language model is properly set for each metric that requires it.
This step is crucial for metrics that need direct access to the model for evaluation purposes.

### Example

```python
def set_model_for_metrics(self):
    """
    Sets the language model for each metric that requires it.
    """
```

In this method, the `Evaluator` class iterates through each metric in the `metrics` list. If a metric has a `set_model`
method, it is called with the `model` attribute, ensuring that each metric can access the necessary language model.

## Evaluation Process

The `evaluate` method performs the evaluation using the provided metrics. It iterates through each metric, performs the
evaluation, and stores the results in a dictionary. The method includes error handling to ensure that any issues
encountered during evaluation are logged.

### Example

```python
def evaluate(self):
    """
    Evaluates the language model using the provided metrics and returns the results.

    Returns:
        dict: A dictionary containing the evaluation results for each metric.
    """
    results = {}
    for metric in self.metrics:
        metric_name = metric.__class__.__name__
        try:
            logger.info(f"Evaluating metric: {metric_name}")
            # Example for Faithfulness metric
            if isinstance(metric, Faithfulness):
                claims = metric.evaluate_claims()
                truths = metric.evaluate_truths()
                verdicts = metric.evaluate_verdicts(claims.claims)
                reason = metric.evaluate_reason(verdicts, truths.truths)
                results['faithfulness'] = {
                    'claims': claims.claims,
                    'truths': truths.truths,
                    'verdicts': [verdict.__dict__ for verdict in verdicts.verdicts],
                    'reason': reason.reason
                }
            # Additional metrics (Answer Relevancy, Bias, etc.) follow a similar pattern.
            logger.info(f"Completed evaluation for metric: {metric_name}")
        except Exception as e:
            logger.error(f"Error evaluating metric {metric_name}: {str(e)}")
    return results
```

In this example, the Evaluator class method evaluate initializes an empty dictionary called results. It iterates through
each metric in the metrics list, retrieves the metric's name, and logs the start of the evaluation process. For each
metric, the method checks its type and calls the appropriate evaluation method. For instance, the Faithfulness metric
involves evaluating claims, truths, and verdicts, which are then stored in the results dictionary under the '
faithfulness' key.

If any exceptions are encountered during the evaluation process, they are caught and logged as errors, and the
evaluation continues with the next metric. The final results are returned as a dictionary containing the outcomes of the
evaluations for each metric.