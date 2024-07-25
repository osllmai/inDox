# KnowledgeRetention

Class for evaluating the retention of knowledge in language model outputs by analyzing the continuity of knowledge across multiple messages, generating verdicts, and calculating retention scores.

## Initialization

The `KnowledgeRetention` class is initialized with the following parameters:

- **messages**: A list of messages containing queries and LLM responses.
- **threshold**: The threshold for determining successful knowledge retention. Defaults to `0.5`.
- **include_reason**: Whether to include reasoning for the knowledge retention verdicts. Defaults to `True`.
- **strict_mode**: Whether to use strict mode, which forces a score of 0 if retention is below the threshold. Defaults to `False`.

```python
class KnowledgeRetention:
    """
    Class for evaluating the retention of knowledge in language model outputs by analyzing the continuity of knowledge
    across multiple messages, generating verdicts, and calculating retention scores.
    """
    def __init__(self, messages: List[Dict[str, str]], threshold: float = 0.5, include_reason: bool = True, strict_mode: bool = False):
        """
        Initializes the KnowledgeRetention class with the messages, threshold, and evaluation settings.

        Parameters:
        messages (List[Dict[str, str]]): A list of messages containing queries and LLM responses.
        threshold (float): The threshold for determining successful knowledge retention. Defaults to 0.5.
        include_reason (bool): Whether to include reasoning for the knowledge retention verdicts. Defaults to True.
        strict_mode (bool): Whether to use strict mode, which forces a score of 0 if retention is below the threshold. Defaults to False.
        """
```
# Hyperparameters Explanation

- **messages**: A list of dictionaries, where each dictionary contains a query and the corresponding `llm_response`. This allows for evaluation of how well the language model retains knowledge across multiple interactions.

- **threshold**: A float value representing the minimum retention score required for knowledge retention to be considered successful. The default value is 0.5.

- **include_reason**: A boolean that indicates whether to provide detailed reasoning for the retention score verdicts. Default is True.

- **strict_mode**: A boolean that, when set to True, forces a score of 0 if the retention score is below the threshold. This is useful for strict evaluation criteria. Default is False.

# Usage Example

Here is an example of how to use the `KnowledgeRetention` class:

```python
import os
from dotenv import load_dotenv
from indox.IndoxEval.llms import OpenAi
from indox.IndoxEval import KnowledgeRetention, Evaluator

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the language model
llm = OpenAi(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

# Define the messages containing queries and LLM responses
messages = [
    {"query": "What is the capital of France?", "llm_response": "The capital of France is Paris."},
    {"query": "Where is the Eiffel Tower located?", "llm_response": "The Eiffel Tower is located in Berlin."}
]

# Initialize the KnowledgeRetention evaluation metric
knowledge_retention_metric = KnowledgeRetention(
    messages=messages, 
    threshold=0.5, 
    include_reason=True, 
    strict_mode=False
)

# Create an evaluator with the KnowledgeRetention metric
evaluator = Evaluator(model=llm, metrics=[knowledge_retention_metric])
result = evaluator.evaluate()
```
