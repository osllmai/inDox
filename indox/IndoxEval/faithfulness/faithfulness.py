from typing import List, Tuple
from pydantic import BaseModel, Field
import json

from indox.IndoxEval.faithfulness.template import FaithfulnessTemplate


class FaithfulnessVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[FaithfulnessVerdict]


class Truths(BaseModel):
    truths: List[str]


class Claims(BaseModel):
    claims: List[str]


class Reason(BaseModel):
    reason: str


class Faithfulness:
    def __init__(self, llm_response, retrieval_context):
        self.model = None
        self.llm_response = llm_response
        self.retrieval_context = retrieval_context

    def set_model(self, model):
        self.model = model

    def evaluate_claims(self) -> Claims:
        prompt = FaithfulnessTemplate.generate_claims(self.llm_response)
        response = self._call_language_model(prompt)
        claims = json.loads(response).get('claims', [])
        return Claims(claims=claims)

    def evaluate_truths(self) -> Truths:
        prompt = FaithfulnessTemplate.generate_truths(self.llm_response)
        response = self._call_language_model(prompt)
        truths = json.loads(response).get('truths', [])
        return Truths(truths=truths)

    def evaluate_verdicts(self, claims: List[str]) -> Verdicts:
        prompt = FaithfulnessTemplate.generate_verdicts(claims, self.retrieval_context)
        response = self._call_language_model(prompt)
        verdicts = json.loads(response).get('verdicts', [])
        return Verdicts(verdicts=[FaithfulnessVerdict(**verdict) for verdict in verdicts])

    def evaluate_reason(self, score: float, contradictions: List[str]) -> Reason:
        prompt = FaithfulnessTemplate.generate_reason(score, contradictions)
        response = self._call_language_model(prompt)
        reason = json.loads(response).get('reason', '')
        return Reason(reason=reason)

    def _call_language_model(self, prompt: str) -> str:
        response = self.model.generate_evaluation_response(prompt=prompt)
        return response
