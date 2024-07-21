import abc
from pydantic import BaseModel

class BaseLLM( BaseModel , abc.ABC):

    @abc.abstractmethod
    def answer_question(self, context, question, max_tokens=200):
        pass

    @abc.abstractmethod
    def get_summary(self, documentation):
        pass

    @abc.abstractmethod
    def grade_docs(self, context, question):
        pass

    @abc.abstractmethod
    def check_hallucination(self, context, answer):
        pass

    class config:
        arbitrary_types_allowed = True

