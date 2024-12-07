import json
from typing import List
from pydantic import BaseModel, Field
from .template import RobustnessTemplate
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


class RobustnessVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)
    score: float = Field(default=0.0)


class Reason(BaseModel):
    reason: str


class Verdicts(BaseModel):
    verdicts: List[RobustnessVerdict]


class AdversarialRobustness:
    def __init__(self, input_sentence: str):
        self.model = None
        self.template = RobustnessTemplate()
        self.input_sentence = input_sentence
        self.robustness_score = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def set_model(self, model):
        self.model = model

    def get_robustness(self) -> List[str]:
        prompt = self.template.generate_reason(self.input_sentence)
        response = self._call_language_model(prompt)
        try:
            data = json.loads(response)
            if data["score"] > 0:
                return [data["reason"]]
            return []
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return []

    def get_reason(self) -> Reason:
        prompt = self.template.generate_reason(self.input_sentence)
        response = self._call_language_model(prompt)
        try:
            data = json.loads(response)
            return Reason(reason=data["reason"])
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return Reason(reason="Error in generating reason.")

    def get_verdict(self) -> RobustnessVerdict:
        prompt = self.template.generate_reason(self.input_sentence)
        response = self._call_language_model(prompt)
        try:
            data = json.loads(response)
            return RobustnessVerdict(
                verdict="yes" if data["score"] > 0.2 else "no",
                reason=data.get("reason", "No reason provided"),
                score=data["score"],
            )
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return RobustnessVerdict(
                verdict="error", reason="Error in generating verdict.", score=0.0
            )

    def calculate_robustness_score(self) -> float:
        verdict = self.get_verdict()
        self.robustness_score = verdict.score
        return self.robustness_score

    def _clean_json_response(self, response: str) -> str:
        """
        Cleans the JSON response from the language model by removing markdown code blocks if present.

        :param response: Raw response from the language model
        :return: Cleaned JSON string
        """
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
            f"Token Counts - Input: {input_token_count} | Output: {output_token_count} | Total: {input_token_count + output_token_count}"
        )

        return clean_response
