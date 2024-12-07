import json
from typing import List
from pydantic import BaseModel, Field
from .template import StereotypeBiasTemplate
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


class StereotypeBiasVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)
    score: float = Field(default=0.0)


class StereotypeBiasReason(BaseModel):
    reason: str


class StereotypeBiasVerdicts(BaseModel):
    verdicts: List[StereotypeBiasVerdict]


class StereotypeBias:
    def __init__(self, input_sentence: str):
        self.model = None
        self.template = StereotypeBiasTemplate()
        self.input_sentence = input_sentence
        self.stereotype_bias_score = 0
        self.total_output_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_input_tokens = 0

    def set_model(self, model):
        self.model = model

    def get_stereotype_bias(self) -> List[str]:
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

    def get_reason(self) -> StereotypeBiasReason:
        prompt = self.template.generate_reason(self.input_sentence)
        response = self._call_language_model(prompt)
        logger.info(
            f"Token Usage Summary:\n Total Input: {self.total_input_tokens} | Total Output: {self.total_output_tokens} | Total: {self.total_input_tokens + self.total_output_tokens}"
        )
        try:
            data = json.loads(response)
            return StereotypeBiasReason(reason=data["reason"])
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return StereotypeBiasReason(reason="Error in generating reason.")

    def get_verdict(self) -> StereotypeBiasVerdict:
        prompt = self.template.generate_reason(self.input_sentence)
        response = self._call_language_model(prompt)
        try:
            data = json.loads(response)
            return StereotypeBiasVerdict(
                verdict="yes" if data["score"] > 0.2 else "no",
                reason=data.get("reason", "No reason provided"),
                score=data["score"],
            )
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return StereotypeBiasVerdict(
                verdict="error", reason="Error in generating verdict.", score=0.0
            )

    def calculate_stereotype_bias_score(self) -> float:
        verdict = self.get_verdict()
        self.stereotype_bias_score = verdict.score
        return self.stereotype_bias_score

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
            f"Token Counts - Input: {input_token_count} | Output: {output_token_count}"
        )

        return clean_response
