# ContextualRelevancy

Class for evaluating the contextual relevancy of retrieval contexts based on a given query using a specified language model.

## Initialization

The `ContextualRelevancy` class is initialized with the following parameters:

- **query**: The query being evaluated.
- **retrieval_context**: A list of contexts retrieved for the query.

```python
class ContextualRelevancy:
    """
    Class for evaluating the contextual relevancy of retrieval contexts based on a given query
    using a specified language model.
    """
    def __init__(self, query: str, retrieval_context: List[str]):
        """
        Initializes the ContextualRelevancy class with the query and retrieval contexts.

        :param query: The query being evaluated.
        :param retrieval_context: A list of contexts retrieved for the query.
        """
```

# Usage Example

Here is an example of how to use the `ContextualRelevancy` class:

```python
import os
from dotenv import load_dotenv
from indoxJudge.models import OpenAi
from indoxJudge.pipelines import Evaluator
from indoxJudge.metrics import ContextualRelevancy

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the language model
# it can be any OpenAI model, please refer to the [OpenAI Models documentation](https://platform.openai.com/docs/models) such as GPT-4o.

llm = OpenAi(api_key=OPENAI_API_KEY, model="Open AI Model")

# Define the query and the retrieval contexts to be evaluated
query = "What are the main causes of global warming?"
retrieval_context = [
    "Human activities, such as burning fossil fuels, deforestation, and industrial processes, are major contributors.",
    "Natural factors, including volcanic eruptions and variations in solar radiation, also play a role.",
    "The greenhouse effect, driven by the accumulation of greenhouse gases like CO2, is a key mechanism."
]

# Initialize the ContextualRelevancy metric
contextual_relevancy_metric = ContextualRelevancy(query=query, retrieval_context=retrieval_context)
evaluator = Evaluator(model=llm, metrics=[contextual_relevancy_metric])
result = evaluator.judge()
```
