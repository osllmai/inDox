# Evaluator

## Overview

The `Evaluator` class provides a comprehensive framework for evaluating language model outputs across multiple dimensions. It supports a wide range of metrics including Faithfulness, Answer Relevancy, Bias, Contextual Relevancy, Hallucination, Knowledge Retention, Toxicity, and various text quality metrics like BertScore, BLEU, and METEOR. This class is essential for conducting thorough assessments of language model performance.

## Initialization

The `Evaluator` class is initialized with the following components:

- **Model**: The language model to be evaluated.
- **Metrics**: A list of metric instances to evaluate the model.

```python
class Evaluator:
    def __init__(self, model, metrics: List):
        """
        Initializes the Evaluator with a language model and a list of metrics.

        Args:
            model: The language model to be evaluated.
            metrics (List): A list of metric instances to evaluate the model.
        """
        self.model = model
        self.metrics = metrics
        logger.info("Evaluator initialized with model and metrics.")
        self.set_model_for_metrics()
        self.evaluation_score = 0
        self.metrics_score = {}
        self.results = {}
```

## Setting the Model for Metrics

The `set_model_for_metrics` method ensures that each metric has access to the language model for evaluation.

```python
def set_model_for_metrics(self):
    """
    Sets the language model for each metric that requires it.
    """
    for metric in self.metrics:
        if hasattr(metric, "set_model"):
            metric.set_model(self.model)
    logger.info("Model set for all metrics.")
```

## Evaluation Process

The `judge` method performs the evaluation using all provided metrics and returns comprehensive results. It processes each metric individually, handling their specific evaluation requirements and aggregating the results.

```python
def judge(self):
    """
    Evaluates the language model using the provided metrics and returns the results.

    Returns:
        dict: A dictionary containing the evaluation results for each metric.
    """
```

### Supported Metrics

The Evaluator supports numerous metrics, each with specific evaluation approaches:

#### Faithfulness

Evaluates factual accuracy by analyzing claims, truths, and providing verdicts with reasoning.

#### Answer Relevancy

Measures how relevant the model's responses are to the given queries.

#### Knowledge Retention

Assesses how well the model retains and applies information provided in context.

#### Hallucination

Identifies when the model generates information not supported by the context.

#### Toxicity

Detects harmful or offensive content in the model's outputs.

#### Bias

Identifies various types of bias in the model's responses.

#### Quality Metrics

Includes standard NLP evaluation metrics like BertScore, BLEU, and METEOR to assess text quality.

#### Safety Metrics

Includes Fairness, Harmfulness, Privacy, and other safety-oriented evaluations.

#### Robustness Metrics

Evaluates model performance under adversarial conditions or with out-of-distribution inputs.

### Example Result Structure

The results from the evaluation are structured as a dictionary:

```python
{
    "Faithfulness": {
        "claims": [...],
        "truths": [...],
        "verdicts": [...],
        "score": 0.85,
        "reason": "..."
    },
    "AnswerRelevancy": {
        "score": 0.92,
        "reason": "...",
        "statements": [...],
        "verdicts": [...]
    },
    # Other metrics follow similar patterns
}
```

## Visualization

The Evaluator provides built-in visualization capabilities through the `plot` method:

```python
def plot(self, mode="external"):
    """
    Visualizes the evaluation results.

    Args:
        mode (str): Visualization mode, either "external" or "inline".

    Returns:
        Visualization object: The visualization of evaluation results.
    """
```

This method creates a visual representation of the evaluation results, making it easier to interpret model performance across different metrics.
