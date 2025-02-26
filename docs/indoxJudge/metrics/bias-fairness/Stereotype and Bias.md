**StereotypeBias**

**Overview**  
Detects specific stereotypical patterns in text outputs related to gender, race, occupation, and cultural associations. Part of the **Bias & Fairness** metric category.

```python
from indoxJudge.metrics import StereotypeBias

# Initialize with text to analyze
analyzer = StereotypeBias(input_sentence="Your text here")
```

**Key Characteristics**  
**Property** | **Description**
--- | ---
**Detection Scope** | Gender, racial, occupational, and cultural stereotypes
**Score Range** | 0.0 (neutral) - 1.0 (severe bias)
**Response Format** | Returns structured verdict with score, flags, and explanation
**Dependencies** | Requires language model integration via `set_model()`

**Interpretation Guide**  
**Score Range** | **Interpretation**
--- | ---
0.0-0.3 | Minimal/no detectable bias
0.3-0.6 | Potential stereotypical patterns
0.6-0.8 | Clear biased representations
0.8-1.0 | Severe harmful stereotypes

**Usage Example**

```python
from indoxJudge.metrics import StereotypeBias
from indoxJudge.pipelines import Evaluator

text = "Elderly workers should avoid technical roles"

# Initialize analyzer
bias_check = StereotypeBias(input_sentence=text)

# Use in evaluation pipeline
evaluator = Evaluator(
    model=your_model,
    metrics=[bias_check]
)

results = evaluator.judge()

# Access full report
print(f"""
Bias Score: {results['stereotype_bias']['score']:.2f}
Explanation: {results['stereotype_bias']['reason']}
""")
```

**Configuration Options**  
**Parameter** | **Effect**
--- | ---
`threshold=0.7` | Adjust sensitivity for flagging (default: 0.7)
Custom Templates | Override default prompts for specific bias detection needs

**Best Practices**

1. **Combine with Context**: Use alongside `ContextualRelevancy` for situational analysis
2. **Threshold Tuning**: Lower threshold (0.5-0.6) for high-risk applications
3. **Cultural Calibration**: Supplement with locale-specific dictionaries
4. **Model Alignment**: Verify detection patterns match your LLM's training data

**Comparison Table**  
**Metric** | **Focus Area** | **Granularity** | **Output Type**
--- | --- | --- | ---
`StereotypeBias` | Specific stereotype detection | Fine-grained | Score + Flags
`Bias` | General bias patterns | Coarse | Binary
`Fairness` | Outcome equality | Statistical | Probability

**Limitations**

1. **Cultural Blindspots**: May under-detect region-specific stereotypes
2. **Context Sensitivity**: Requires clear referents for accurate detection
3. **Nuance Gap**: Struggles with sarcasm/irony in stereotype expression
4. **Model Dependency**: Effectiveness tied to underlying LLM capabilities

**Error Handling**  
**Common Issues** | **Recommended Action**
--- | ---
Low confidence scores | Simplify sentence structure
Multiple stereotype types | Use `Bias` metric for broad analysis
Ambiguous references | Provide additional context
API failures | Implement retry logic (3 attempts recommended)
