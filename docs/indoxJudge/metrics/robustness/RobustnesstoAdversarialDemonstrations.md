**Robustness To Adversarial Demonstrations**

**Overview**  
Evaluates a language model's resilience against misleading or manipulative demonstrations by measuring its ability to maintain reliable performance and appropriate behavior. Part of the **Security & Defense** metric category.

```python
from indoxJudge.metrics import RobustnessToAdversarialDemonstrations

# Initialize with text to analyze
adversarial_check = RobustnessToAdversarialDemonstrations(input_sentence="Your text here")
```

**Key Characteristics**  
**Property** | **Description**
--- | ---
**Detection Scope** | Manipulation attempts, misleading examples, deceptive demonstrations
**Score Range** | 0.0 (vulnerable) - 1.0 (highly robust)
**Response Format** | Returns robustness score with vulnerability assessment and defense recommendations
**Dependencies** | Requires language model integration via `set_model()`

**Interpretation Guide**  
**Score Range** | **Interpretation**
--- | ---
0.0-0.2 | Highly vulnerable to adversarial influence
0.2-0.4 | Substantial susceptibility to manipulation
0.4-0.6 | Moderate resilience with some vulnerabilities
0.6-0.8 | Good resistance to most adversarial demonstrations
0.8-1.0 | Excellent robustness against manipulation

**Usage Example**

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

# Access robustness assessment
print(f"""
Adversarial Robustness Score: {results['adversarial_robustness']['score']:.2f}
Vulnerabilities Detected: {results['adversarial_robustness']['reason']}
""")
```

**Best Practices**

1. **Diverse Attack Simulation**: Test against multiple adversarial techniques (prompt injection, social engineering)
2. **Context-Aware Evaluation**: Consider domain-specific manipulation risks
3. **Boundary Testing**: Assess edge cases where policy enforcement is ambiguous
4. **Progressive Difficulty**: Start with simple manipulations and increase sophistication

**Limitations**

1. **Evolving Tactics**: New adversarial techniques may emerge that aren't covered
2. **False Confidence**: High scores don't guarantee protection against novel attacks
3. **Context Blindness**: May miss contextual nuances that enable sophisticated attacks
4. **Static Templates**: Fixed evaluation strategies may not reflect dynamic attack landscapes

**Error Handling**  
**Common Issues** | **Recommended Action**
--- | ---
Invalid model responses | Implement retry with different prompts
JSON parsing errors | Use robust parsing with exception handling
Template rendering issues | Verify template compatibility with model
Invalid input formats | Normalize and sanitize inputs before evaluation

## Flow Chart

```mermaid
flowchart TD
    A[Initialize Adversarial Robustness] --> B[Set Input Sentence]
    B --> C[Set Model]
    C --> D[Generate Adversarial Examples]
    D --> E[Test Model Responses]
    E --> F[Evaluate Response Appropriateness]
    F --> G[Identify Vulnerability Patterns]
    G --> H[Calculate Robustness Score]
    H --> I[Generate Defense Recommendations]
    I --> J[Return Comprehensive Assessment]
```
