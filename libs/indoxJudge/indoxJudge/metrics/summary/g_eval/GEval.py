from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from loguru import logger
import json
from .GEvalTemplate import GEvalTemplate


class GrammarVerdict(BaseModel):
    aspect: str
    score: float
    reason: str = Field(default=None)
    issues_found: Optional[List[str]] = None


class GrammarScores(BaseModel):
    scores: List[GrammarVerdict]


class GEval:
    def __init__(
        self,
        summary: str,
        include_reason: bool = True,
        weights: Dict[str, float] = None,
    ):
        """Initialize GEval metric for grammar evaluation.

        Args:
            summary: Text to evaluate
            include_reason: Whether to include detailed reasons in output
            weights: Custom weights for different aspects of evaluation
        """
        self.summary = summary
        self.include_reason = include_reason
        self.weights = weights or {
            "grammar_correctness": 0.35,
            "sentence_structure": 0.25,
            "coherence": 0.25,
            "readability": 0.15,
        }
        self.model = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def set_model(self, model):
        """Set the language model to use for evaluation."""
        self.model = model

    def measure(self) -> Dict:
        """Measure grammatical quality of the text.

        Returns:
            Dictionary containing scores, issues, and analysis
        """
        if not self.model:
            raise ValueError("Model not set. Please call set_model() first.")

        self.grammar_issues = self._extract_grammar_issues()
        self.grammar_scores = self._evaluate_grammar()
        self.score = self._calculate_weighted_score()
        self.issue_distribution = self._analyze_issue_distribution()

        if self.include_reason:
            self.verdict = self._generate_final_verdict()

        logger.info(
            f"Token Usage Summary:\n Total Input: {self.total_input_tokens} | "
            f"Total Output: {self.total_output_tokens} | "
            f"Total: {self.total_input_tokens + self.total_output_tokens}"
        )

        result = {
            "score": round(self.score, 3),
            "grammar_issues": self.grammar_issues,
            "grammar_scores": [score.dict() for score in self.grammar_scores],
            "issue_distribution": self.issue_distribution,
        }

        if self.include_reason:
            result["verdict"] = self.verdict

        return result

    def _extract_grammar_issues(self) -> List[str]:
        """Extract specific grammar issues from the text."""
        prompt = GEvalTemplate.extract_grammar_issues(text=self.summary)
        response = self._call_language_model(prompt)
        data = json.loads(response)
        return data["grammar_issues"]

    def _evaluate_grammar(self) -> List[GrammarVerdict]:
        """Evaluate different aspects of grammatical quality."""
        prompt = GEvalTemplate.evaluate_grammar(summary=self.summary)
        response = self._call_language_model(prompt)
        data = json.loads(response)
        return [GrammarVerdict(**item) for item in data["scores"]]

    def _analyze_issue_distribution(self) -> Dict[str, int]:
        """Analyze the distribution of grammar issues across categories."""
        distribution = {}
        grammar_verdict = next(
            (
                score
                for score in self.grammar_scores
                if score.aspect.lower().replace(" ", "_") == "grammar_correctness"
            ),
            None,
        )

        if grammar_verdict and grammar_verdict.issues_found:
            for issue_type in self.grammar_issues:
                distribution[issue_type] = sum(
                    1
                    for found in grammar_verdict.issues_found
                    if issue_type.lower() in found.lower()
                )
        return distribution

    def _calculate_weighted_score(self) -> float:
        """Calculate final weighted score based on individual aspect scores."""
        total_score = 0.0
        total_weight = 0.0

        for verdict in self.grammar_scores:
            aspect_name = verdict.aspect.lower().replace(" ", "_")
            weight = self.weights.get(aspect_name, 0.0)
            total_score += verdict.score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_score / total_weight

    def _generate_final_verdict(self) -> str:
        """Generate a final verdict summarizing the grammatical analysis."""
        scores_dict = [score.dict() for score in self.grammar_scores]
        prompt = GEvalTemplate.generate_final_verdict(
            scores=scores_dict,
            final_score=self.score,
            issue_distribution=self.issue_distribution,
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
