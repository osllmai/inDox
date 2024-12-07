from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from loguru import logger
import json
from .relevanceTemplate import RelevanceTemplate


class RelevanceVerdict(BaseModel):
    aspect: str
    score: float
    reason: str = Field(default=None)
    key_points_matched: Optional[List[str]] = None


class RelevanceScores(BaseModel):
    scores: List[RelevanceVerdict]


class Relevance:
    def __init__(
        self,
        summary: str,
        source_text: str,
        include_reason: bool = True,
        weights: Dict[str, float] = None,
    ):
        self.summary = summary
        self.source_text = source_text
        self.include_reason = include_reason
        self.weights = weights or {
            "key_information_coverage": 0.4,
            "topic_alignment": 0.3,
            "information_accuracy": 0.2,
            "focus_distribution": 0.1,
        }
        self.model = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def set_model(self, model):
        self.model = model

    def measure(self) -> float:
        self.key_points = self._extract_key_points()
        self.relevance_scores = self._evaluate_relevance()
        self.score = self._calculate_weighted_score()
        self.key_points_coverage = self._analyze_key_points_coverage()

        if self.include_reason:
            self.verdict = self._generate_final_verdict()

        logger.info(
            f"Token Usage Summary:\n Total Input: {self.total_input_tokens} | "
            f"Total Output: {self.total_output_tokens} | "
            f"Total: {self.total_input_tokens + self.total_output_tokens}"
        )
        return {
            "score": round(self.score, 3),
            "key_points": self.key_points,
            "relevance_scores": self.relevance_scores,
            "key_point_coverage": self.key_points_coverage,
        }

    def _extract_key_points(self) -> List[str]:
        prompt = RelevanceTemplate.extract_key_points(text=self.source_text)
        response = self._call_language_model(prompt)
        data = json.loads(response)
        return data["key_points"]

    def _evaluate_relevance(self) -> List[RelevanceVerdict]:
        prompt = RelevanceTemplate.evaluate_relevance(
            summary=self.summary, source_text=self.source_text
        )
        response = self._call_language_model(prompt)
        data = json.loads(response)
        return [RelevanceVerdict(**item) for item in data["scores"]]

    def _analyze_key_points_coverage(self) -> Dict[str, bool]:
        coverage = {}
        key_info_verdict = next(
            (
                score
                for score in self.relevance_scores
                if score.aspect.lower().replace(" ", "_") == "key_information_coverage"
            ),
            None,
        )

        if key_info_verdict and key_info_verdict.key_points_matched:
            for point in self.key_points:
                coverage[point] = point in key_info_verdict.key_points_matched
        return coverage

    def _calculate_weighted_score(self) -> float:
        total_score = 0.0
        for verdict in self.relevance_scores:
            aspect_name = verdict.aspect.lower().replace(" ", "_")
            weight = self.weights.get(aspect_name, 0.25)
            total_score += verdict.score * weight
        return total_score

    def _generate_final_verdict(self) -> str:
        scores_dict = [score.dict() for score in self.relevance_scores]
        prompt = RelevanceTemplate.generate_final_verdict(
            scores=scores_dict,
            final_score=self.score,
            key_points_coverage=self.key_points_coverage,
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
