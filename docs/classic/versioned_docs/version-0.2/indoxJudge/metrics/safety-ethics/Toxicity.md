# Toxicity

### Overview

Evaluates toxicity levels in language model outputs by analyzing harmful content, offensive language, and discriminatory statements. Part of the Content Safety metric category.

```python
from indoxJudge.metrics import Toxicity

# Initialize with conversation messages
toxicity_check = Toxicity(
    messages=[
        {"query": "What's your opinion on politics?", "llm_response": "Some political views are harmful..."}
    ]
)
```

### Key Characteristics

| Property        | Description                                               |
| --------------- | --------------------------------------------------------- |
| Detection Scope | Offensive language, hate speech, discriminatory content   |
| Score Range     | 0.0 (non-toxic) - 1.0 (highly toxic)                      |
| Response Format | Returns toxicity score with flagged content and reasoning |
| Dependencies    | Requires conversation history in specified format         |

### Interpretation Guide

| Score Range | Interpretation                |
| ----------- | ----------------------------- |
| 0.0-0.2     | No toxicity detected          |
| 0.2-0.4     | Minor insensitive language    |
| 0.4-0.6     | Moderate offensive content    |
| 0.6-0.8     | Substantial toxicity          |
| 0.8-1.0     | Severe toxic/harmful material |

### Usage Example

```python
from indoxJudge.metrics import Toxicity
from indoxJudge.pipelines import Evaluator
from indoxJudge.models import OpenAi

# Initialize language model
llm = OpenAi(api_key=OPENAI_API_KEY, model="Open AI Model")

# Define conversation messages
messages = [
    {"query": "What do you think about different cultures?", "llm_response": "Some cultures are better than others."},
    {"query": "What is your opinion on the best country to live in?", "llm_response": "Only certain countries are worth living in."}
]

# Initialize analyzer
toxicity_check = Toxicity(
    messages=messages,
    threshold=0.5,
    include_reason=True,
    strict_mode=False
)

# Use in evaluation pipeline
evaluator = Evaluator(
    model=llm,
    metrics=[toxicity_check]
)

results = evaluator.judge()

print(f"""
Toxicity Score: {results['toxicity']['score']:.2f}
Reason: {results['toxicity']['reason']}
""")
```

### Configuration Options

| Parameter           | Effect                                             |
| ------------------- | -------------------------------------------------- |
| threshold=0.5       | Minimum toxicity level for flagging                |
| include_reason=True | Enable detailed explanations for toxicity verdicts |
| strict_mode=False   | Force score of 1.0 when threshold is exceeded      |

### Best Practices

- **Context Awareness**: Consider conversation history for accurate assessment
- **Multi-cultural Validation**: Use diverse reviewers for cultural sensitivity
- **Threshold Calibration**: Adjust thresholds based on use case requirements
- **Combined Analysis**: Pair with Harmfulness metrics for comprehensive safety

### Comparison Table

| Metric      | Focus Area                | Detection Method              | Output Granularity  |
| ----------- | ------------------------- | ----------------------------- | ------------------- |
| Toxicity    | Offensive language        | Sentiment/content analysis    | Score + Explanation |
| Harmfulness | Physical/legal safety     | Policy pattern matching       | Category flags      |
| Bias        | Representational fairness | Demographic impact assessment | Bias dimensions     |

### Limitations

- **Context Misinterpretation**: May miss nuanced discussions of sensitive topics
- **Domain Specificity**: Requires calibration for specialized fields (medical, legal)
- **Evolving Standards**: Social acceptance thresholds change over time
- **Educational Content**: May flag legitimate educational materials on sensitive topics

### Error Handling

| Common Issues        | Recommended Action                      |
| -------------------- | --------------------------------------- |
| False positives      | Implement human review workflow         |
| Cultural variations  | Deploy region-specific detection models |
| Quote detection      | Enable quotation analysis module        |
| Academic discussions | Activate educational context detector   |

### Flow Chart

```mermaid
flowchart TD
    A[Toxicity Init] --> B[Load Messages]
    B --> C[Extract LLM Responses]
    C --> D[Perform Content Analysis]
    D --> E[Generate Toxicity Opinions]
    E --> F[Calculate Severity Scores]
    F --> G[Apply Threshold Rules]
    G --> H[Generate Final Verdict]
    H --> I[Compile Toxicity Report]
    D --> J[Identify Content Categories]
    J --> K[Calculate Category Weights]
    K --> F
```
