from typing import List, Optional, Union
from pydantic import BaseModel, Field
import asyncio
import json

from indox.eval.answer_relevancy.template import AnswerRelevancyTemplate


class Statements(BaseModel):
    statements: List[str]


class AnswerRelevancyVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[AnswerRelevancyVerdict]


class Reason(BaseModel):
    reason: str


class AnswerRelevancyMetric:
    def __init__(self, threshold: float = 0.5, include_reason: bool = True, strict_mode: bool = False):
        self.threshold = 1 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.evaluation_cost = None

    def measure(self, input: str, actual_output: str) -> float:
        self.statements = self._generate_statements(actual_output)
        self.verdicts = self._generate_verdicts(input)
        self.score = self._calculate_score()
        self.reason = self._generate_reason(input)
        self.success = self.score >= self.threshold

        return self.score

    def _generate_statements(self, actual_output: str) -> List[str]:
        prompt = AnswerRelevancyTemplate.generate_statements(actual_output=actual_output)
        response = self._mock_model_generate(prompt)
        data = json.loads(response)
        return data["statements"]

    def _generate_verdicts(self, input: str) -> List[AnswerRelevancyVerdict]:
        prompt = AnswerRelevancyTemplate.generate_verdicts(input=input, actual_output=self.statements)
        response = self._mock_model_generate(prompt)
        data = json.loads(response)
        return [AnswerRelevancyVerdict(**item) for item in data["verdicts"]]

    def _generate_reason(self, input: str) -> str:
        if not self.include_reason:
            return None

        irrelevant_statements = [verdict.reason for verdict in self.verdicts if verdict.verdict.strip().lower() == "no"]

        prompt = AnswerRelevancyTemplate.generate_reason(
            irrelevant_statements=irrelevant_statements,
            input=input,
            score=format(self.score, ".2f"),
        )

        response = self._mock_model_generate(prompt)
        data = json.loads(response)
        return data["reason"]

    def _calculate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 1

        relevant_count = sum(1 for verdict in self.verdicts if verdict.verdict.strip().lower() != "no")

        score = relevant_count / number_of_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

    def _mock_model_generate(self, prompt: str) -> str:
        # This is a mock function simulating an LLM response. Replace this with actual model call.
        print(prompt)  # For debugging, you can see the prompt being generated
        if "generate_statements" in prompt:
            return json.dumps({"statements": ["Example statement 1.", "Example statement 2."]})
        elif "generate_verdicts" in prompt:
            return json.dumps({
                "verdicts": [
                    {"verdict": "yes"},
                    {"verdict": "no", "reason": "Example reason for irrelevance."}
                ]
            })
        elif "generate_reason" in prompt:
            return json.dumps({"reason": "The score is 0.50 because one of the statements was irrelevant."})


# Example Usage
input_text = "What should I do if there is an earthquake?"
actual_output = "Shoes. Thanks for asking the question! Is there anything else I can help you with? Duck and hide"

metric = AnswerRelevancyMetric()
score = metric.measure(input=input_text, actual_output=actual_output)
print(f"Relevancy Score: {score}")
print(f"Reason: {metric.reason}")
