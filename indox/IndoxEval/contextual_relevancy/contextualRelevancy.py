import json
from typing import List
from pydantic import BaseModel, Field
from .template import ContextualRelevancyTemplate


class ContextualRelevancyVerdict(BaseModel):
    """
    Model representing a verdict on the contextual relevancy of a query and retrieval context,
    including the verdict itself and the reasoning behind it.
    """
    verdict: str
    reason: str = Field(default=None)


class Verdicts(BaseModel):
    """
    Model representing a list of ContextualRelevancyVerdict instances.
    """
    verdicts: List[ContextualRelevancyVerdict]


class Reason(BaseModel):
    """
    Model representing the reason provided for irrelevancies in the retrieval context.
    """
    reason: str


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
        self.model = None
        self.template = ContextualRelevancyTemplate()
        self.query = query
        self.retrieval_contexts = retrieval_context
        self.irrelevancies = []
        self.score = 0

    def set_model(self, model):
        """
        Sets the language model to be used for evaluation.

        :param model: The language model to use.
        """
        self.model = model

    def get_irrelevancies(self, query: str, retrieval_contexts: List[str]) -> List[str]:
        """
        Evaluates the retrieval contexts for irrelevancies based on the query.

        :param query: The query being evaluated.
        :param retrieval_contexts: A list of contexts to evaluate.
        :return: A list of irrelevancies found in the contexts.
        """
        irrelevancies = []
        for retrieval_context in retrieval_contexts:
            prompt = self.template.generate_verdict(query, retrieval_context)
            response = self._call_language_model(prompt=prompt)
            data = json.loads(response)
            if data["verdict"].strip().lower() == "no":
                irrelevancies.append(data["reason"])
        return irrelevancies

    def set_irrelevancies(self, irrelevancies: List[str]):
        """
        Sets the list of irrelevancies found during evaluation.

        :param irrelevancies: A list of irrelevancies.
        """
        self.irrelevancies = irrelevancies

    def get_reason(self, irrelevancies: List[str], score: float) -> Reason:
        """
        Generates the reasoning behind the irrelevancies and score.

        :param irrelevancies: A list of irrelevancies.
        :param score: The score assigned to the evaluation.
        :return: A Reason object containing the reasoning.
        """
        prompt = self.template.generate_reason(self.query, irrelevancies, score)
        response = self._call_language_model(prompt=prompt)
        data = json.loads(response)
        return Reason(reason=data["reason"])

    def get_verdict(self, query: str, retrieval_context: str) -> ContextualRelevancyVerdict:
        """
        Evaluates the contextual relevancy verdict for a single retrieval context.

        :param query: The query being evaluated.
        :param retrieval_context: A single context to evaluate.
        :return: A ContextualRelevancyVerdict object containing the verdict and reason.
        """
        prompt = self.template.generate_verdict(query=query, context=retrieval_context)
        response = self._call_language_model(prompt=prompt)
        data = json.loads(response)
        return ContextualRelevancyVerdict(verdict=data["verdict"], reason=data.get("reason", "No reason provided"))

    def get_verdicts(self, query: str, retrieval_contexts: List[str]) -> Verdicts:
        """
        Evaluates the contextual relevancy verdicts for multiple retrieval contexts.

        :param query: The query being evaluated.
        :param retrieval_contexts: A list of contexts to evaluate.
        :return: A Verdicts object containing the list of verdicts.
        """
        verdicts = [self.get_verdict(query, retrieval_context) for retrieval_context in retrieval_contexts]
        return Verdicts(verdicts=verdicts)

    def _call_language_model(self, prompt: str) -> str:
        """
        Calls the language model with the given prompt and returns the response.

        :param prompt: The prompt to provide to the language model.
        :return: The response from the language model.
        """
        response = self.model.generate_evaluation_response(prompt=prompt)
        return response
