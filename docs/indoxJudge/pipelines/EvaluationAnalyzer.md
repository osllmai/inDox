# LLMComparison

## Overview

The `LLMComparison` class is designed to facilitate the comparison of multiple language models based on their evaluation metrics. This class allows for visual representation of the performance of different models, making it easier to analyze and compare their strengths and weaknesses.

## Initialization

The `LLMComparison` class is initialized with a list of models, where each model is represented by a dictionary containing its name, overall score, and various evaluation metrics.

### Example

```python
class LLMComparison:
    def __init__(self, models):
        """
        Initializes the LLMComparison with a list of models.

        Args:
            models (list): A list of dictionaries where each dictionary contains
                           the model's name, score, and metrics.
        """
```

## Plotting the Comparison

The plot method generates a visual representation of the comparison between the models. It uses an external visualization library to create graphs that illustrate the performance of each model across various metrics. The mode parameter allows for different modes of visualization.

## Usage Example

Below is an example of how to use the LLMComparison class to compare different language models and generate a plot.

```python
from indoxJudge.pipelines import EvaluationAnalyzer

models = [
    {
        'name': 'Model_1',
        'score': 0.50,
        'metrics': {
            'Faithfulness': 0.55,
            'AnswerRelevancy': 1.0,
            'Bias': 0.45,
            'Hallucination': 0.8,
            'KnowledgeRetention': 0.0,
            'Toxicity': 0.0,
            'precision': 0.64,
            'recall': 0.77,
            'f1_score': 0.70,
            'BLEU': 0.11
        }
    },
    {
        'name': 'Model_2',
        'score': 0.61,
        'metrics': {
            'Faithfulness': 1.0,
            'AnswerRelevancy': 1.0,
            'Bias': 0.0,
            'Hallucination': 0.8,
            'KnowledgeRetention': 1.0,
            'Toxicity': 0.0,
            'precision': 0.667,
            'recall': 0.77,
            'f1_score': 0.71,
            'BLEU': 0.14
        }
    },
    {
        'name': 'Model_3',
        'score': 0.050,
        'metrics': {
            'Faithfulness': 1.0,
            'AnswerRelevancy': 1.0,
            'Bias': 0.0,
            'Hallucination': 0.83,
            'KnowledgeRetention': 0.0,
            'Toxicity': 0.0,
            'precision': 0.64,
            'recall': 0.76,
            'f1_score': 0.70,
            'BLEU': 0.10
        }
    }
]

llm_comparison = EvaluationAnalyzer(models=models)
llm_comparison.plot(mode="inline")
```

In this example, three models are compared based on their metrics, and the results are plotted inline. The LLMComparison class facilitates the easy comparison and visualization of different language models' performances.
