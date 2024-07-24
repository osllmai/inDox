from typing import List, Optional, Union
from pydantic import BaseModel, Field
import json

from indox.IndoxEval.answer_relevancy.template import AnswerRelevancyTemplate


class Statements(BaseModel):
    statements: List[str]


class AnswerRelevancyVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[AnswerRelevancyVerdict]


class Reason(BaseModel):
    reason: str


class AnswerRelevancy:
    def __init__(self, query, llm_response, model=None, threshold: float = 0.5, include_reason: bool = True,
                 strict_mode: bool = False):
        self.model = model
        self.query = query
        self.llm_response = llm_response
        self.threshold = 1 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.evaluation_cost = None
        self.query = None
        self.llm_response = None
        self.statements = []
        self.verdicts = []
        self.reason = None
        self.score = 0
        self.success = False

    def set_model(self, model):
        self.model = model

    def measure(self) -> float:

        self.statements = self._generate_statements()
        self.verdicts = self._generate_verdicts()
        self.score = self._calculate_score()
        self.reason = self._generate_reason(self.query)
        self.success = self.score >= self.threshold

        return self.score

    def _generate_statements(self) -> List[str]:
        prompt = AnswerRelevancyTemplate.generate_statements(llm_response=self.llm_response)
        response = self._call_language_model(prompt)
        data = json.loads(response)
        return data["statements"]

    def _generate_verdicts(self) -> List[AnswerRelevancyVerdict]:
        prompt = AnswerRelevancyTemplate.generate_verdicts(query=self.query, llm_response=self.statements)
        response = self._call_language_model(prompt)
        data = json.loads(response)
        return [AnswerRelevancyVerdict(**item) for item in data["verdicts"]]

    def _generate_reason(self, query: str) -> str:
        if not self.include_reason:
            return None

        irrelevant_statements = [verdict.reason for verdict in self.verdicts if verdict.verdict.strip().lower() == "no"]

        prompt = AnswerRelevancyTemplate.generate_reason(
            irrelevant_statements=irrelevant_statements,
            query=query,
            score=format(self.score, ".2f"),
        )

        response = self._call_language_model(prompt)
        data = json.loads(response)
        return data["reason"]

    def _calculate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 1

        relevant_count = sum(1 for verdict in self.verdicts if verdict.verdict.strip().lower() != "no")

        score = relevant_count / number_of_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

    def _call_language_model(self, prompt: str) -> str:
        response = self.model.generate_evaluation_response(prompt=prompt)
        return response

