from typing import List, Dict, Union
from pydantic import BaseModel, Field


class Knowledge(BaseModel):
    data: Dict[str, Union[str, List[str]]]


class KnowledgeRetentionVerdict(BaseModel):
    index: int
    verdict: str
    reason: str = Field(default=None)


import json
from typing import List, Dict, Union


class KnowledgeRetentionMetric:
    def __init__(self, threshold: float = 0.5, include_reason: bool = True, strict_mode: bool = False):
        self.threshold = 1 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode

    def measure(self, test_case) -> float:
        self.knowledges = self._generate_knowledges(test_case)
        self.verdicts = self._generate_verdicts(test_case)
        knowledge_retention_score = self._calculate_score()
        self.reason = self._generate_reason(knowledge_retention_score)
        self.success = knowledge_retention_score >= self.threshold
        self.score = knowledge_retention_score
        return self.score

    def _generate_reason(self, score: float) -> str:
        if not self.include_reason:
            return None

        attritions = [verdict.reason for verdict in self.verdicts if verdict.verdict.strip().lower() == "yes"]

        prompt = KnowledgeRetentionTemplate.generate_reason(
            attritions=attritions,
            score=format(score, ".2f"),
        )

        response = self._mock_model_generate(prompt)
        data = json.loads(response)
        return data["reason"]

    def _calculate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 0

        retention_count = sum(1 for verdict in self.verdicts if verdict.verdict.strip().lower() == "no")

        score = retention_count / number_of_verdicts

        return 0 if self.strict_mode and score < self.threshold else score

    def _generate_verdicts(self, test_case) -> List[KnowledgeRetentionVerdict]:
        verdicts = []
        for index, message in enumerate(test_case["messages"]):
            previous_knowledge = self.knowledges[index].data

            prompt = KnowledgeRetentionTemplate.generate_verdict(
                llm_message=message["actual_output"],
                previous_knowledge=previous_knowledge,
            )
            response = self._mock_model_generate(prompt)
            data = json.loads(response)
            verdict = KnowledgeRetentionVerdict(index=index, **data)
            verdicts.append(verdict)

        return verdicts

    def _generate_knowledges(self, test_case) -> List[Knowledge]:
        knowledges = []
        for index, message in enumerate(test_case["messages"]):
            previous_knowledge = knowledges[-1].data if knowledges else {}
            llm_message = test_case["messages"][index - 1]["actual_output"] if index > 0 else ""

            prompt = KnowledgeRetentionTemplate.extract_data(
                llm_message=llm_message,
                user_message=message["input"],
                previous_knowledge=previous_knowledge,
            )

            response = self._mock_model_generate(prompt)
            data = json.loads(response)
            knowledge = Knowledge(data=data)
            knowledges.append(knowledge)

        return knowledges

    def _mock_model_generate(self, prompt: str) -> str:
        # This is a mock function simulating an LLM response. Replace this with an actual model call.
        print(prompt)  # For debugging, you can see the prompt being generated
        if "generate_reason" in prompt:
            return json.dumps(
                {"reason": "The score is 0.50 because some information was forgotten from previous knowledge."})
        elif "generate_verdict" in prompt:
            if "London for holiday" in prompt:
                return json.dumps({"verdict": "yes",
                                   "reason": "The LLM suggests the user has already been to London for holiday when it was a work trip instead."})
            elif "where do you live" in prompt:
                return json.dumps({"verdict": "yes",
                                   "reason": "The LLM is asking where the user lives when the address of the user is already known to be '83 Belvedere, London' from earlier in the conversation."})
            else:
                return json.dumps({"verdict": "no"})
        elif "extract_data" in prompt:
            if "Jimmy Carter" in prompt:
                return json.dumps({"39th President of USA": "Jimmy Carter"})
            elif "1989" in prompt:
                return json.dumps({"Birthday": "January 21st 1989"})
            elif "Jennifer" in prompt:
                return json.dumps({"Names of sisters-in-law": ["Jennifer", "Mandy"]})
            elif "Model X" in prompt:
                return json.dumps({"Tesla Model": "Model X"})
            else:
                return json.dumps({})


# Example Usage
test_case = {
    "messages": [
        {"input": "What is the name of the 39th President of the USA?",
         "actual_output": "The 39th President of the USA is Jimmy Carter."},
        {"input": "When were you born?", "actual_output": "I was born on January 21st 1989."},
        {"input": "Do you have any sisters-in-law?", "actual_output": "Yes, I have a sister-in-law named Jennifer."},
        {"input": "Which model of Tesla do you drive?", "actual_output": "I drive a Tesla Model X."}
    ]
}

metric = KnowledgeRetentionMetric()
score = metric.measure(test_case=test_case)
print(f"Knowledge Retention Score: {score}")
print(f"Reason: {metric.reason}")
