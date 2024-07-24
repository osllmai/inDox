import json
from typing import List
from pydantic import BaseModel, Field

from indox.IndoxEval.hallucination.template import HallucinationTemplate


class HallucinationVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[HallucinationVerdict]


class Reason(BaseModel):
    reason: str


class Hallucination:
    def __init__(self, llm_response, retrieval_context, threshold: float = 0.5, include_reason: bool = True,
                 strict_mode: bool = False):
        self.threshold = 0 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.evaluation_cost = None
        self.llm_response = llm_response
        self.retrieval_context = retrieval_context
        self.model = None

    def set_model(self, model):
        self.model = model

    def measure(self) -> float:
        self.verdicts = self._generate_verdicts()
        self.score = self._calculate_score()
        self.reason = self._generate_reason()
        self.success = self.score <= self.threshold

        return self.score

    def _generate_verdicts(self) -> List[HallucinationVerdict]:
        prompt = HallucinationTemplate.generate_verdicts(actual_output=self.llm_response,
                                                         contexts=self.retrieval_context)
        response = self._call_language_model(prompt)
        data = json.loads(response)
        return [HallucinationVerdict(**item) for item in data["verdicts"]]

    def _generate_reason(self) -> str:
        if not self.include_reason:
            return None

        factual_alignments = []
        contradictions = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)
            else:
                factual_alignments.append(verdict.reason)

        prompt = HallucinationTemplate.generate_reason(
            factual_alignments=factual_alignments,
            contradictions=contradictions,
            score=format(self.score, ".2f"),
        )

        response = self._call_language_model(prompt)
        data = json.loads(response)
        return data["reason"]

    def _calculate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 0

        hallucination_count = sum(1 for verdict in self.verdicts if verdict.verdict.strip().lower() == "no")

        score = hallucination_count / number_of_verdicts
        return 1 if self.strict_mode and score > self.threshold else score

    def _call_language_model(self, prompt: str) -> str:
        response = self.model.generate_evaluation_response(prompt=prompt)
        return response
