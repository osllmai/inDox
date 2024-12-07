import json
import re
from typing import List
from pydantic import BaseModel, Field
from .template import ContextualRelevancyTemplate
from loguru import logger
import sys

# Set up logging
logger.remove()  # Remove the default logger
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)

logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


class ContextualRelevancyVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[ContextualRelevancyVerdict]


class Reason(BaseModel):
    reason: str


class ContextualRelevancy:
    def __init__(self, query: str, retrieval_context: List[str]):
        """Initialize with query and retrieval contexts."""
        self.model = None
        self.template = ContextualRelevancyTemplate()
        self.query = query
        self.retrieval_contexts = retrieval_context
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def set_model(self, model):
        """Set the language model to be used for evaluation."""
        self.model = model

    def set_irrelevancies(self, irrelevancies: List[str]):
        self.irrelevancies = irrelevancies

    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response by removing markdown code blocks if present."""
        response = response.strip()
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

        clean_response = self._clean_json_response(response)
        output_token_count = len(enc.encode(response))
        self.total_output_tokens += output_token_count
        logger.info(
            f"Token Counts - Input: {input_token_count} | Output: {output_token_count}"
        )
        return clean_response

    def get_verdict(
        self, query: str, retrieval_context: str
    ) -> ContextualRelevancyVerdict:
        """Get verdict for a single context."""
        prompt = self.template.generate_verdict(query=query, context=retrieval_context)
        response = self._call_language_model(prompt)

        try:
            # Remove single quotes between words to handle potential JSON issues
            response = re.sub(r"(?<=\w)'(?=\w)", "", response)
            data = json.loads(response)

            if "verdict" not in data:
                raise ValueError("Missing 'verdict' key in the model's response.")

            return ContextualRelevancyVerdict(
                verdict=data["verdict"], reason=data.get("reason", "No reason provided")
            )
        except json.JSONDecodeError:
            return ContextualRelevancyVerdict(
                verdict="error", reason="Error in generating verdict."
            )
        except Exception:
            return ContextualRelevancyVerdict(verdict="error", reason="Invalid format.")

    def get_verdicts(self, query: str, retrieval_contexts: List[str]) -> Verdicts:
        """Get verdicts for all contexts."""
        verdicts = [
            self.get_verdict(query, retrieval_context)
            for retrieval_context in retrieval_contexts
        ]
        return Verdicts(verdicts=verdicts)

    def get_irrelevancies(self, query: str, retrieval_contexts: List[str]) -> List[str]:
        """Get list of reasons why contexts are irrelevant."""
        irrelevancies = []
        for retrieval_context in retrieval_contexts:
            prompt = self.template.generate_verdict(query, retrieval_context)
            response = self._call_language_model(prompt)

            try:
                data = json.loads(response.strip())
                if data["verdict"].strip().lower() == "no":
                    irrelevancies.append(data["reason"])
            except (json.JSONDecodeError, KeyError, Exception) as e:
                logger.error(f"Error processing verdict: {e}")
                continue

        return irrelevancies

    def get_reason(self, irrelevancies: List[str], score: float) -> Reason:
        """Get final reasoning based on irrelevancies and score."""
        prompt = self.template.generate_reason(self.query, irrelevancies, score)
        response = self._call_language_model(prompt)

        try:
            data = json.loads(response)
            return Reason(reason=data["reason"])
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            return Reason(reason="Error in generating reason.")

    def measure(self) -> dict:
        """
        Complete evaluation flow that matches the external usage pattern.
        """
        # Get irrelevancies
        irrelevancies = self.get_irrelevancies(self.query, self.retrieval_contexts)

        # Get verdicts
        verdicts = self.get_verdicts(self.query, self.retrieval_contexts)
        # Calculate score based on irrelevancies
        score = (
            1.0
            if not irrelevancies
            else max(0, 1.0 - len(irrelevancies) / len(self.retrieval_contexts))
        )

        # Get final reasoning
        reason = self.get_reason(irrelevancies, score)
        logger.info(
            f"Token Usage Summary:\n Total Input: {self.total_input_tokens} | Total Output: {self.total_output_tokens} | Total: {self.total_input_tokens + self.total_output_tokens}"
        )
        return {
            "verdicts": [verdict.dict() for verdict in verdicts.verdicts],
            "reason": reason.dict(),
            "score": round(score, 2),
        }
