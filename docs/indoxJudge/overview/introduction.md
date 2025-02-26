# IndoxJudge Evaluation Framework

## Overview

IndoxJudge is a comprehensive evaluation framework for language model applications, designed to systematically assess various aspects of AI-generated content through configurable pipelines and metrics.

### Key Capabilities

- **Multi-Dimensional Evaluation**: Analyze outputs across 6 core dimensions:

  - Basic NLP Quality
  - Contextual Relevance
  - Safety & Ethics
  - Bias & Fairness
  - Informational Accuracy
  - Robustness

- **Pipeline-Driven Architecture**: Preconfigured evaluation workflows:

  ```python
  __all__ = [
      "Evaluator",          # Base Evaluator that takes custom list of metrics
      "LLMEvaluator",       # General language model evaluation
      "RagEvaluator",       # Retrieval-Augmented Generation systems
      "SafetyEvaluator",    # Content safety analysis
      "SummaryEvaluator",   # Summarization-specific checks
      "EvaluationAnalyzer"  # Compare multiple LLMs
  ]
  ```

- **Extensive Metric Library**: 40+ specialized metrics across categories:

  ```python
  # Basic NLP Metrics
  "BLEU", "METEOR", "Rouge", "BertScore", "Gruen", "GEval"

  # Relevancy and Context Metrics
  "AnswerRelevancy", "ContextualRelevancy", "KnowledgeRetention", "Faithfulness"

  # Safety and Ethics Metrics
  "Toxicity", "ToxicityDiscriminative", "SafetyToxicity", "Harmfulness",
  "MachineEthics", "Privacy"

  # Bias and Fairness Metrics
  "Bias", "Fairness", "StereotypeBias"

  # Quality and Accuracy Metrics
  "Hallucination", "Misinformation"

  # Robustness Metrics
  "AdversarialRobustness", "OutOfDistributionRobustness",
  "RobustnessToAdversarialDemonstrations"

  # Summary-Specific Metrics
  "SummaryBertScore", "SummaryBleu", "Conciseness", "FactualConsistency",
  "SummaryGEval", "InformationCoverage", "SummaryMeteor", "Relevance",
  "SummaryRouge", "StructureQuality", "SummaryToxicity"
  ```

## Why IndoxJudge?

- **Holistic Assessment**: Combine lexical, semantic, and ethical evaluations
- **Modular Design**: Mix-and-match metrics for custom evaluation workflows
- **Actionable Insights**: Detailed scoring with explanatory feedback
- **Enterprise-Ready**: Scalable architecture for batch processing

## Getting Started

```python
from indoxJudge.pipelines import Evaluator
from indoxJudge.metrics import Toxicity, AnswerRelevancy

evaluator = LLMEvaluator(
    model=your_llm,
    metrics=[Toxicity(), AnswerRelevancy()]
)

results = evaluator.judge("Sample LLM output")
```
