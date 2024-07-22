import json
from typing import List
from pydantic import BaseModel, Field
from .template import ContextualRelevancyTemplate


class ContextualRelevancyVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[ContextualRelevancyVerdict]


class Reason(BaseModel):
    reason: str


class ContextualRelevancy:
    def __init__(self, query: str, retrieval_context: List[str]):
        self.model = None
        self.template = ContextualRelevancyTemplate()
        self.query = query
        self.retrieval_contexts = retrieval_context
        self.irrelevancies = []
        self.score = 0

    def set_model(self, model):
        self.model = model

    def get_irrelevancies(self, query: str, retrieval_contexts: List[str]) -> List[str]:
        irrelevancies = []
        for retrieval_context in retrieval_contexts:
            prompt = self.template.generate_verdict(query, retrieval_context)
            response = self._call_language_model(prompt=prompt)
            data = json.loads(response)
            if data["verdict"].strip().lower() == "no":
                irrelevancies.append(data["reason"])
        return irrelevancies

    def set_irrelevancies(self, irrelevancies: List[str]):
        self.irrelevancies = irrelevancies

    def get_reason(self, irrelevancies: List[str], score: float) -> Reason:
        prompt = self.template.generate_reason(self.query, irrelevancies, score)
        response = self._call_language_model(prompt=prompt)
        data = json.loads(response)
        return Reason(reason=data["reason"])

    def get_verdict(self, query: str, retrieval_context: str) -> ContextualRelevancyVerdict:
        prompt = self.template.generate_verdict(query=query,context= retrieval_context)
        response = self._call_language_model(prompt=prompt)
        data = json.loads(response)
        print(data)
        return ContextualRelevancyVerdict(verdict=data["verdict"], reason=data.get("reason", "No reason provided"))

    def get_verdicts(self, query: str, retrieval_contexts: List[str]) -> Verdicts:
        verdicts = [self.get_verdict(query, retrieval_context) for retrieval_context in retrieval_contexts]
        return Verdicts(verdicts=verdicts)

    def _call_language_model(self, prompt: str) -> str:
        response = self.model.generate_evaluation_response(prompt=prompt)
        return response