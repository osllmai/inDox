from dspy import Signature, Module, ChainOfThought, Prediction, InputField, OutputField, OpenAI, settings
import os


class GenerateAnswer(Signature):
    """
    Signature class for generating an answer.

    Attributes:
        context (InputField): Contextual information helpful for answering the question.
        question (InputField): The question to be answered.
        answer (OutputField): The generated answer.
    """
    context = InputField(desc="Helpful information for answering the question.")
    question = InputField()
    answer = OutputField(desc="A detailed answer that is supported by the context. ONLY OUTPUT THE ANSWER!!")


class RAG(Module):
    """
    Module class for RAG (Retrieval-Augmented Generation) model.

    Attributes:
        generate_answer (ChainOfThought): Chain of thought for generating an answer.
        model (str): The model to be used.
        client (OpenAI): The OpenAI client.
    """

    def __init__(self, model, client):
        super().__init__()
        self.generate_answer = ChainOfThought(GenerateAnswer)
        self.model = model
        self.client = client

    def forward(self, context, question):
        """
        Forward method for the RAG model.

        Args:
            context (str): The context for the question.
            question (str): The question to be answered.

        Returns:
            Prediction: The prediction containing the context, question, and answer.
        """
        response = self.generate_answer(context=context, question=question, model=self.model, client=self.client)
        return Prediction(context=context, question=question, answer=response.answer)


class DspyCotQA(Module):
    """
    Module class for DspyCotQA.

    Attributes:
        model (str): The model to be used.
        client (OpenAI): The OpenAI client.
    """

    def __init__(self, model, api_key):
        super().__init__()
        self.model = model
        self.client = OpenAI(model=self.model, api_key=api_key)
        settings.configure(lm=self.client)

    def answer_question(self, context, question):
        """
        Answers a question given a context using the specified model.

        Args:
            context (str): The context for the question.
            question (str): The question to be answered.

        Returns:
            str: The answer to the question.
        """
        rag = RAG(model=self.model, client=self.client)
        prediction = rag.forward(context, question)
        return prediction.answer
