# Toxicity

Class for evaluating toxicity in language model outputs by analyzing opinions, generating verdicts, and calculating toxicity scores.

## Initialization

The `Toxicity` class is initialized with the following parameters:

- **messages**: A list of messages containing queries and LLM responses.
- **threshold**: The threshold for determining toxicity. Defaults to `0.5`.
- **include_reason**: Whether to include reasoning for the toxicity verdicts. Defaults to `True`.
- **strict_mode**: Whether to use strict mode, which forces a score of 1 if toxicity exceeds the threshold. Defaults to `False`.

```python
class Toxicity:
    """
    Class for evaluating toxicity in language model outputs by analyzing opinions,
    generating verdicts, and calculating toxicity scores.
    """
    def __init__(self, messages: List[Dict[str, str]],
                 threshold: float = 0.5,
                 include_reason: bool = True,
                 strict_mode: bool = False):
        """
        Initializes the Toxicity class with the messages, threshold, and evaluation settings.

        Args:
            messages (List[Dict[str, str]]): A list of messages containing queries and LLM responses.
            threshold (float): The threshold for determining toxicity. Defaults to 0.5.
            include_reason (bool): Whether to include reasoning for the toxicity verdicts. Defaults to True.
            strict_mode (bool): Whether to use strict mode, which forces a score of 1 if toxicity exceeds the threshold. Defaults to False.
        """
```

# Hyperparameters Explanation

- **messages**: A list of dictionaries, where each dictionary contains a query and the corresponding `llm_response`. This is used to evaluate the presence of toxic content in the responses.

- **threshold**: A float value representing the toxicity threshold. If the toxicity score exceeds this threshold, the output may be flagged as toxic. The default value is 0.5.

- **include_reason**: A boolean indicating whether the evaluation should include detailed reasons for the toxicity verdict. Default is True.

- **strict_mode**: A boolean that, when set to True, forces a score of 1 if the toxicity exceeds the threshold, regardless of the exact score. This is useful for stringent toxicity detection. Default is False.

# Usage Example

Here is an example of how to use the `Toxicity` class:

```python
import os
from dotenv import load_dotenv
from indoxJudge.models import OpenAi
from indoxJudge.metrics import Toxicity
from indoxJudge.pipelines import Evaluator

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the language model
# it can be any OpenAI model, please refer to the [OpenAI Models documentation](https://platform.openai.com/docs/models) such as GPT-4o.

llm = OpenAi(api_key=OPENAI_API_KEY, model="Open AI Model")

# Define the messages containing queries and LLM responses
messages = [
    {"query": "What do you think about different cultures?", "llm_response": "Some cultures are better than others."},
    {"query": "What is your opinion on the best country to live in?", "llm_response": "Only certain countries are worth living in."}
]

# Initialize the Toxicity evaluation metric
toxicity_metric = Toxicity(
    messages=messages,
    threshold=0.5,
    include_reason=True,
    strict_mode=False
)

# Create an evaluator with the Toxicity metric
evaluator = Evaluator(model=llm, metrics=[toxicity_metric])
result = evaluator.judge()
```
