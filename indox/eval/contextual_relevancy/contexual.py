from typing import List
from pydantic import BaseModel, Field
from .template import ContextualRelevancyTemplate
import os


# Schema definition
class ContextualRelevancyVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[ContextualRelevancyVerdict]


class Reason(BaseModel):
    reason: str


# Functionality for ContextualRelevancy
class ContextualRelevancy:
    def __init__(self, template: ContextualRelevancyTemplate):
        self.template = template

    def get_reason(self, input: str, irrelevancies: str, score: float) -> Reason:
        prompt = self.template.generate_reason(input, irrelevancies, score)
        from indox.eval.llms import OpenAi
        llm = OpenAi(api_key=os.getenv("OPENAI_API"), model="gpt-3.5-turbo-0125")
        response = llm.generate_evaluation_response(prompt=prompt)
        return Reason(response)

    def get_verdict(self, text: str, context: str) -> ContextualRelevancyVerdict:
        prompt = self.template.generate_verdict(text, context)
        from indox.eval.llms import OpenAi
        llm = OpenAi(api_key=os.getenv("OPENAI_API"), model="gpt-3.5-turbo-0125")
        response = llm.generate_evaluation_response(prompt=prompt)
        return ContextualRelevancyVerdict(response)

    def get_verdicts(self, texts: List[str], contexts: List[str]) -> Verdicts:
        verdicts = [self.get_verdict(text, context) for text, context in zip(texts, contexts)]
        return Verdicts(verdicts=verdicts)


# Example usage
template = ContextualRelevancyTemplate()
contextual_relevancy = ContextualRelevancy(template)

# Example data
input_text = "What are the symptoms of COVID-19?"
irrelevancies_text = "The context mentions historical data on pandemics which is not related to COVID-19 symptoms."
score_value = 0.5
context_text = "The Black Death was one of the most devastating pandemics in human history."

# Generate reason
reason = contextual_relevancy.get_reason(input_text, irrelevancies_text, score_value)
print(reason.json())

# Generate verdict
verdict = contextual_relevancy.get_verdict(input_text, context_text)
print(verdict.json())

# Generate multiple verdicts
texts = ["What is AI?", "Who won the World Cup in 2018?"]
contexts = ["AI stands for Artificial Intelligence.", "The Nobel Prize is awarded annually."]
verdicts = contextual_relevancy.get_verdicts(texts, contexts)
print(verdicts.json())
