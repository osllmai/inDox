from typing import List
from pydantic import BaseModel, Field


class HallucinationVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[HallucinationVerdict]


class Reason(BaseModel):
    reason: str


import json
from typing import List, Optional, Union
from pydantic import BaseModel, Field


class HallucinationMetric:
    def __init__(self, threshold: float = 0.5, include_reason: bool = True, strict_mode: bool = False):
        self.threshold = 0 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.evaluation_cost = None

    def measure(self, contexts: List[str], actual_output: str) -> float:
        self.verdicts = self._generate_verdicts(actual_output, contexts)
        self.score = self._calculate_score()
        self.reason = self._generate_reason()
        self.success = self.score <= self.threshold

        return self.score

    def _generate_verdicts(self, actual_output: str, contexts: List[str]) -> List[HallucinationVerdict]:
        prompt = HallucinationTemplate.generate_verdicts(actual_output=actual_output, contexts=contexts)
        response = self._mock_model_generate(prompt)
        data = json.loads(response)
        return [HallucinationVerdict(**item) for item in data["verdicts"]]

    def _generate_reason(self) -> str:
        if not self.include_reason:
            return None

        factual_alignments = []
        contradictions = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)
            else:
                factual_alignments.append(verdict.reason)

        prompt = HallucinationTemplate.generate_reason(
            factual_alignments=factual_alignments,
            contradictions=contradictions,
            score=format(self.score, ".2f"),
        )

        response = self._mock_model_generate(prompt)
        data = json.loads(response)
        return data["reason"]

    def _calculate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 0

        hallucination_count = sum(1 for verdict in self.verdicts if verdict.verdict.strip().lower() == "no")

        score = hallucination_count / number_of_verdicts
        return 1 if self.strict_mode and score > self.threshold else score

    def _mock_model_generate(self, prompt: str) -> str:
        # This is a mock function simulating an LLM response. Replace this with actual model call.
        print(prompt)  # For debugging, you can see the prompt being generated
        if "generate_verdicts" in prompt:
            return json.dumps({
                "verdicts": [
                    {"verdict": "yes", "reason": "The actual output agrees with the context."},
                    {"verdict": "no", "reason": "The actual output contradicts the context. The correct year is 1968."}
                ]
            })
        elif "generate_reason" in prompt:
            return json.dumps(
                {"reason": "The score is 0.50 because one of the contexts contradicted the actual output."})


# Example Usage
contexts = [
    "Einstein won the Nobel Prize for his discovery of the photoelectric effect.",
    "Einstein won the Nobel Prize in 1968."
]
actual_output = "Einstein won the Nobel Prize in 1969 for his discovery of the photoelectric effect."

metric = HallucinationMetric()
score = metric.measure(contexts=contexts, actual_output=actual_output)
print(f"Hallucination Score: {score}")
print(f"Reason: {metric.reason}")
