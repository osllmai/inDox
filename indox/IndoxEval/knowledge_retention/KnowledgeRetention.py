from typing import List, Dict, Union
from pydantic import BaseModel, Field
import json

from indox.IndoxEval.knowledge_retention.template import KnowledgeRetentionTemplate

class Knowledge(BaseModel):
    data: Dict[str, Union[str, List[str]]]

class KnowledgeRetentionVerdict(BaseModel):
    index: int
    verdict: str
    reason: str = Field(default=None)

class KnowledgeRetention:
    def __init__(self, model, messages: List[Dict[str, str]], threshold: float = 0.5, include_reason: bool = True, strict_mode: bool = False):
        self.model = model
        self.messages = messages
        self.threshold = 1 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.knowledges = []
        self.verdicts = []
        self.reason = None
        self.score = None
        self.success = None

    def measure(self) -> float:
        self.knowledges = self._generate_knowledges()
        self.verdicts = self._generate_verdicts()
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

        response = self._call_language_model(prompt)
        data = json.loads(response)
        return data["reason"]

    def _calculate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 0

        retention_count = sum(1 for verdict in self.verdicts if verdict.verdict.strip().lower() == "no")

        score = retention_count / number_of_verdicts

        return 0 if self.strict_mode and score < self.threshold else score

    def _generate_verdicts(self) -> List[KnowledgeRetentionVerdict]:
        verdicts = []
        for index, message in enumerate(self.messages):
            previous_knowledge = self.knowledges[index].data

            prompt = KnowledgeRetentionTemplate.generate_verdict(
                llm_message=message["llm_response"],
                previous_knowledge=previous_knowledge,
            )
            response = self._call_language_model(prompt)
            data = json.loads(response)
            verdict = KnowledgeRetentionVerdict(index=index, **data)
            verdicts.append(verdict)

        return verdicts

    def _generate_knowledges(self) -> List[Knowledge]:
        knowledges = []
        for index, message in enumerate(self.messages):
            previous_knowledge = knowledges[-1].data if knowledges else {}
            llm_message = self.messages[index - 1]["llm_response"] if index > 0 else ""

            prompt = KnowledgeRetentionTemplate.extract_data(
                llm_message=llm_message,
                user_message=message["query"],
                previous_knowledge=previous_knowledge,
            )

            response = self._call_language_model(prompt)
            data = json.loads(response)
            knowledge = Knowledge(data=data)
            knowledges.append(knowledge)

        return knowledges

    def _call_language_model(self, prompt: str) -> str:
        response = self.model.generate_evaluation_response(prompt=prompt)
        return response
