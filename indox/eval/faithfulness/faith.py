import os
from indox.eval.faithfulness.template import FaithfulnessTemplate
from typing import List, Tuple
from pydantic import BaseModel, Field
import json
from dotenv import load_dotenv

load_dotenv()


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


class FaithfulnessEvaluator:
    # def __init__(self, model):
    #     self.model = model

    def evaluate_claims(self, text: str) -> Claims:
        prompt = FaithfulnessTemplate.generate_claims(text)
        response = self.call_language_model(prompt)
        claims = json.loads(response).get('claims', [])
        return Claims(claims=claims)

    def evaluate_truths(self, text: str) -> Truths:
        prompt = FaithfulnessTemplate.generate_truths(text)
        response = self.call_language_model(prompt)
        truths = json.loads(response).get('truths', [])
        return Truths(truths=truths)

    def evaluate_verdicts(self, claims: List[str], retrieval_context: str) -> Verdicts:
        prompt = FaithfulnessTemplate.generate_verdicts(claims, retrieval_context)
        response = self.call_language_model(prompt)
        verdicts = json.loads(response).get('verdicts', [])
        return Verdicts(verdicts=[FaithfulnessVerdict(**verdict) for verdict in verdicts])

    def evaluate_reason(self, score: float, contradictions: List[str]) -> Reason:
        prompt = FaithfulnessTemplate.generate_reason(score, contradictions)
        response = self.call_language_model(prompt)
        reason = json.loads(response).get('reason', '')
        return Reason(reason=reason)

    async def a_evaluate_claims(self, text: str) -> Claims:
        prompt = FaithfulnessTemplate.generate_claims(text)
        response = await self.a_call_language_model(prompt)
        claims = json.loads(response).get('claims', [])
        return Claims(claims=claims)

    async def a_evaluate_truths(self, text: str) -> Truths:
        prompt = FaithfulnessTemplate.generate_truths(text)
        response = await self.a_call_language_model(prompt)
        truths = json.loads(response).get('truths', [])
        return Truths(truths=truths)

    async def a_evaluate_verdicts(self, claims: List[str], retrieval_context: str) -> Verdicts:
        prompt = FaithfulnessTemplate.generate_verdicts(claims, retrieval_context)
        response = await self.a_call_language_model(prompt)
        verdicts = json.loads(response).get('verdicts', [])
        return Verdicts(verdicts=[FaithfulnessVerdict(**verdict) for verdict in verdicts])

    async def a_evaluate_reason(self, score: float, contradictions: List[str]) -> Reason:
        prompt = FaithfulnessTemplate.generate_reason(score, contradictions)
        response = await self.a_call_language_model(prompt)
        reason = json.loads(response).get('reason', '')
        return Reason(reason=reason)

    def call_language_model(self, prompt: str) -> str:
        from indox.eval.llms import OpenAi
        from indox.llms import IndoxApi
        llm = OpenAi(api_key=os.getenv("OPENAI_API"), model="gpt-3.5-turbo-0125")
        response = llm.generate_evaluation_response(prompt=prompt)
        return response

    async def a_call_language_model(self, prompt: str) -> str:
        # Implement the call to your asynchronous language model here
        return
