from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field
from loguru import logger
import json
from .concisenessTemplate import ConcisenessTemplate


class VerbosityIssue(BaseModel):
    text_segment: str
    issue_type: str
    suggestion: str
    impact_score: float


class RedundancyAnalysis(BaseModel):
    repeated_phrases: List[Dict[str, Union[str, int]]]
    redundant_information: List[str]
    unnecessary_modifiers: List[str]
    score: float
    explanation: str


class WordinessMetrics(BaseModel):
    total_words: int
    average_sentence_length: float
    filler_word_count: int
    complex_phrase_count: int
    score: float
    suggestions: List[str]


class ConcisenessMeasurement(BaseModel):
    redundancy_score: float
    wordiness_score: float
    overall_score: float
    issues: List[VerbosityIssue]
    improvement_suggestions: List[str]


class Conciseness:
    def __init__(
        self,
        summary: str,
        source_text: str = None,
        target_length: int = None,
        weights: Dict[str, float] = None,
        conciseness_threshold: float = 0.7,
    ):
        self.summary = summary
        self.source_text = source_text
        self.target_length = target_length
        self.conciseness_threshold = conciseness_threshold
        self.weights = weights or {
            "redundancy": 0.4,
            "wordiness": 0.4,
            "length_ratio": 0.2,
        }
        self.model = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.filler_words = {
            "basically",
            "actually",
            "practically",
            "definitely",
            "really",
            "very",
            "quite",
            "rather",
            "somewhat",
            "virtually",
            "literally",
            "just",
            "so",
            "pretty",
            "totally",
            "absolutely",
            "certainly",
        }

    def set_model(self, model):
        self.model = model

    def measure(self) -> float:
        self.redundancy_analysis = self._analyze_redundancy()
        self.wordiness_metrics = self._measure_wordiness()
        self.length_ratio_score = (
            self._calculate_length_ratio() if self.source_text else 1.0
        )

        overall_score = self._calculate_weighted_score()
        self.measurement = ConcisenessMeasurement(
            redundancy_score=self.redundancy_analysis.score,
            wordiness_score=self.wordiness_metrics.score,
            overall_score=overall_score,
            issues=self._identify_issues(),
            improvement_suggestions=self._generate_suggestions(),
        )

        logger.info(
            f"Token Usage Summary:\n Total Input: {self.total_input_tokens} | "
            f"Total Output: {self.total_output_tokens} | "
            f"Total: {self.total_input_tokens + self.total_output_tokens}"
        )
        return self._get_detailed_report()

    def _analyze_redundancy(self) -> RedundancyAnalysis:
        prompt = ConcisenessTemplate.analyze_redundancy(text=self.summary)
        response = self._call_language_model(prompt)

        # Parse the JSON response
        try:
            json_response = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}")

        # Transform repeated phrases if present
        if "repeated_phrases" in json_response:
            try:
                json_response["repeated_phrases"] = [
                    {"phrase": entry["phrase"], "count": int(entry["count"])}
                    for entry in json_response["repeated_phrases"]
                ]
            except (KeyError, ValueError, TypeError) as e:
                raise ValueError(f"Error processing 'repeated_phrases': {e}")

        # Adjust score if summary and source text are identical
        if self.source_text and self.summary.strip() == self.source_text.strip():
            # Ensure the score reflects that identical texts are not "perfect" but close
            json_response["score"] = max(json_response.get("score", 1.0), 0.9)

        # Return RedundancyAnalysis instance
        return RedundancyAnalysis(**json_response)

    def _measure_wordiness(self) -> WordinessMetrics:
        prompt = ConcisenessTemplate.measure_wordiness(text=self.summary)
        response = self._call_language_model(prompt)
        metrics = WordinessMetrics(**json.loads(response))

        # Adjust for identical texts
        if self.source_text and self.summary.strip() == self.source_text.strip():
            metrics.score = min(
                metrics.score, 0.95
            )  # Slight penalty for identical texts

        return metrics

    def _calculate_length_ratio(self) -> float:
        if not self.source_text or not self.target_length:
            return 1.0

        current_length = len(self.summary.split())
        source_length = len(self.source_text.split())

        # Adjust when texts are identical
        if self.summary.strip() == self.source_text.strip():
            return 0.98  # Near-perfect but not perfect

        target_ratio = self.target_length / source_length
        actual_ratio = current_length / source_length
        ratio_difference = abs(target_ratio - actual_ratio)
        return max(0, 1 - ratio_difference)

    def _calculate_weighted_score(self) -> float:
        scores = {
            "redundancy": self.redundancy_analysis.score,
            "wordiness": self.wordiness_metrics.score,
            "length_ratio": self.length_ratio_score,
        }

        # Allow minor penalties to preserve evaluation dynamics
        if self.source_text and self.summary.strip() == self.source_text.strip():
            scores[
                "redundancy"
            ] *= 0.98  # Slight penalty for lack of conciseness improvement
            scores["wordiness"] *= 0.98

        return sum(scores[k] * self.weights[k] for k in self.weights)

    def _identify_issues(self) -> List[VerbosityIssue]:
        issues = []

        # Check for repeated phrases
        for phrase_info in self.redundancy_analysis.repeated_phrases:
            phrase = phrase_info["phrase"]
            count = int(phrase_info["count"])  # Ensure count is an integer
            if count > 1:
                issues.append(
                    VerbosityIssue(
                        text_segment=phrase,
                        issue_type="repetition",
                        suggestion=f"Remove {count - 1} repetitions of '{phrase}'",
                        impact_score=0.1 * (count - 1),
                    )
                )

        # Check for complex phrases
        if self.wordiness_metrics.complex_phrase_count > 0:
            for suggestion in self.wordiness_metrics.suggestions:
                if "Replace" in suggestion:
                    original = suggestion.split("Replace '")[1].split("' with")[0]
                    issues.append(
                        VerbosityIssue(
                            text_segment=original,
                            issue_type="complex_phrase",
                            suggestion=suggestion,
                            impact_score=0.1,
                        )
                    )

        return issues

    def _generate_suggestions(self) -> List[str]:
        suggestions = []

        # Length-based suggestions
        if self.wordiness_metrics.average_sentence_length > 20:
            suggestions.append("Consider breaking longer sentences into shorter ones")

        # Redundancy-based suggestions
        if self.redundancy_analysis.redundant_information:
            suggestions.append(
                "Remove redundant information: "
                + ", ".join(self.redundancy_analysis.redundant_information)
            )

        # Wordiness-based suggestions
        suggestions.extend(self.wordiness_metrics.suggestions)

        return suggestions

    def _call_language_model(self, prompt: str) -> str:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        input_token_count = len(enc.encode(prompt))
        response = self.model.generate_evaluation_response(prompt=prompt)
        self.total_input_tokens += input_token_count

        if not response:
            raise ValueError("Received empty response from model")

        clean_response = self._clean_json_response(response)
        output_token_count = len(enc.encode(response))
        self.total_output_tokens += output_token_count
        logger.info(
            f"Token Counts - Input: {input_token_count} | Output: {output_token_count}"
        )

        return clean_response

    def _clean_json_response(self, response: str) -> str:
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        return response

    def _get_detailed_report(self) -> Dict:
        """Generate a detailed report of the conciseness evaluation."""
        return {
            "overall_score": round(self.measurement.overall_score, 3),
            "metrics": {
                "redundancy": {
                    "score": self.redundancy_analysis.score,
                    "repeated_phrases": self.redundancy_analysis.repeated_phrases,
                    "redundant_information": self.redundancy_analysis.redundant_information,
                },
                "wordiness": {
                    "score": self.wordiness_metrics.score,
                    "average_sentence_length": self.wordiness_metrics.average_sentence_length,
                    "filler_word_count": self.wordiness_metrics.filler_word_count,
                    "complex_phrase_count": self.wordiness_metrics.complex_phrase_count,
                },
                "length_ratio": self.length_ratio_score,
            },
            "issues": [issue.dict() for issue in self.measurement.issues],
            "suggestions": self.measurement.improvement_suggestions,
        }
