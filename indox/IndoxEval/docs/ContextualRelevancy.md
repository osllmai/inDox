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
