from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from loguru import logger
import json
from .toxicityTemplate import ToxicityTemplate


class ToxicityVerdict(BaseModel):
    aspect: str
    score: float
    reason: str = Field(default=None)
    examples_found: Optional[List[str]] = None


class ToxicityScores(BaseModel):
    scores: List[ToxicityVerdict]


class Toxicity:
    def __init__(
        self,
        summary: str,
        include_reason: bool = True,
        weights: Dict[str, float] = None,
    ):
        """Initialize Toxicity metric for content evaluation.

        Args:
            summary: Text to evaluate
            include_reason: Whether to include detailed reasons in output
            weights: Custom weights for different aspects of toxicity
        """
        self.summary = summary
        self.include_reason = include_reason
        self.weights = weights or {
            "hate_speech": 0.35,
            "profanity": 0.25,
            "personal_attacks": 0.25,
            "threat_level": 0.15,
        }
        self.model = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def set_model(self, model):
        """Set the language model to use for evaluation."""
        self.model = model

    def measure(self) -> Dict:
        """Measure toxicity levels in the text.

        Returns:
            Dictionary containing scores, toxic elements, and analysis
        """
        if not self.model:
            raise ValueError("Model not set. Please call set_model() first.")

        self.toxic_elements = self._identify_toxic_elements()
        self.toxicity_scores = self._evaluate_toxicity()
        self.score = self._calculate_weighted_score()
        self.element_distribution = self._analyze_element_distribution()

        if self.include_reason:
            self.verdict = self._generate_final_verdict()

        logger.info(
            f"Token Usage Summary:\n Total Input: {self.total_input_tokens} | "
            f"Total Output: {self.total_output_tokens} | "
            f"Total: {self.total_input_tokens + self.total_output_tokens}"
        )

        result = {
            "score": round(self.score, 3),
            "toxic_elements": self.toxic_elements,
            "toxicity_scores": [score.dict() for score in self.toxicity_scores],
            "element_distribution": self.element_distribution,
        }

        if self.include_reason:
            result["verdict"] = self.verdict

        return result

    def _identify_toxic_elements(self) -> List[str]:
        """Identify specific toxic elements in the text."""
        prompt = ToxicityTemplate.identify_toxic_elements(text=self.summary)
        response = self._call_language_model(prompt)
        data = json.loads(response)
        return data["toxic_elements"]

    def _evaluate_toxicity(self) -> List[ToxicityVerdict]:
        """Evaluate different aspects of toxicity."""
        prompt = ToxicityTemplate.evaluate_toxicity(text=self.summary)
        response = self._call_language_model(prompt)
        data = json.loads(response)
        return [ToxicityVerdict(**item) for item in data["scores"]]

    def _analyze_element_distribution(self) -> Dict[str, int]:
        """Analyze the distribution of toxic elements across categories."""
        distribution = {}
        for score in self.toxicity_scores:
            if hasattr(score, "examples_found") and score.examples_found:
                aspect_name = score.aspect.lower().replace(" ", "_")
                distribution[aspect_name] = len(score.examples_found)
        return distribution

    def _calculate_weighted_score(self) -> float:
        """Calculate final weighted score based on individual aspect scores."""
        total_score = 0.0
        total_weight = 0.0

        for verdict in self.toxicity_scores:
            aspect_name = verdict.aspect.lower().replace(" ", "_")
            weight = self.weights.get(aspect_name, 0.0)
            total_score += verdict.score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_score / total_weight

    def _generate_final_verdict(self) -> str:
        """Generate a final verdict summarizing the toxicity analysis."""
        scores_dict = [score.dict() for score in self.toxicity_scores]
        prompt = ToxicityTemplate.generate_final_verdict(
            scores=scores_dict,
            final_score=self.score,
            element_distribution=self.element_distribution,
        )
        response = self._call_language_model(prompt)
        data = json.loads(response)
        return data["verdict"]

    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response from language model."""
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        return response

    def _call_language_model(self, prompt: str) -> str:
        """Call the language model and track token usage."""
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
