from typing import List, Dict
from pydantic import BaseModel, Field
from loguru import logger
import json
import sys
from .structureQualityTemplate import StructureTemplate


class StructureVerdict(BaseModel):
    aspect: str
    score: float
    reason: str = Field(default=None)


class StructureScores(BaseModel):
    scores: List[StructureVerdict]


class StructureQuality:
    def __init__(
        self,
        summary: str,
        weights: Dict[str, float] = None,
    ):
        self.summary = summary
        self.weights = weights or {
            "discourse_coherence": 0.3,
            "logical_flow": 0.3,
            "topic_consistency": 0.2,
            "temporal_consistency": 0.2,
        }
        self.model = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def set_model(self, model):
        self.model = model

    def measure(self) -> float:
        self.structure_scores = self._evaluate_structure()
        self.score = self._calculate_weighted_score()
        self.verdict = self._generate_final_verdict()
        logger.info(
            f"Token Usage Summary:\n Total Input: {self.total_input_tokens} | "
            f"Total Output: {self.total_output_tokens} | "
            f"Total: {self.total_input_tokens + self.total_output_tokens}"
        )
        return {
            "score": round(self.score, 3),
            "verdicts": self.structure_scores,
            "reason": self.verdict,
        }

    def _evaluate_structure(self) -> List[StructureVerdict]:
        prompt = StructureTemplate.evaluate_structure(summary=self.summary)
        response = self._call_language_model(prompt)
        data = json.loads(response)
        return [StructureVerdict(**item) for item in data["scores"]]

    def _calculate_weighted_score(self) -> float:
        total_score = 0.0
        for verdict in self.structure_scores:
            aspect_name = verdict.aspect.lower().replace(" ", "_")
            weight = self.weights.get(aspect_name, 0.25)
            total_score += verdict.score * weight
        return total_score

    def _generate_final_verdict(self) -> str:
        scores_dict = [score.dict() for score in self.structure_scores]
        prompt = StructureTemplate.generate_final_verdict(
            scores=scores_dict, final_score=self.score
        )
        response = self._call_language_model(prompt)
        data = json.loads(response)
        return data["verdict"]

    def _clean_json_response(self, response: str) -> str:
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        return response

    def _call_language_model(self, prompt: str) -> str:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        input_token_count = len(enc.encode(prompt))
        response = self.model.generate_evaluation_response(prompt=prompt)
        self.total_input_tokens += input_token_count

        if not response:
            raise ValueError("Received an empty response from the model.")

        clean_response = self._clean_json_response(response=response)
        output_token_count = len(enc.encode(response))
        self.total_output_tokens += output_token_count
        logger.info(
            f"Token Counts - Input: {input_token_count} | Output: {output_token_count}"
        )

        return clean_response
