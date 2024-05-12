from dspy import Signature, Module, ChainOfThought, Prediction, InputField, OutputField, OpenAI, settings
import os


class GenerateAnswer(Signature):
    """Assess the context and answer the question."""
    context = InputField(desc="Helpful information for answering the question.")
    question = InputField()
    answer = OutputField(desc="A detailed answer that is supported by the context. ONLY OUTPUT THE ANSWER!!")


class RAG(Module):
    def __init__(self, model, client):
        super().__init__()
        self.generate_answer = ChainOfThought(GenerateAnswer)
        self.model = model
        self.client = client

    def forward(self, context, question):
        response = self.generate_answer(context=context, question=question, model=self.model, client=self.client)
        return Prediction(context=context, question=question, answer=response.answer)


class DspyCotQA(Module):
    def __init__(self, model="gpt-3.5-turbo-0125", api_key=os.environ["OPENAI_API_KEY"]):
        super().__init__()
        self.model = model
        self.client = OpenAI(model=self.model, api_key=api_key)
        settings.configure(lm=self.client)

    def answer_question(self, context, question):
        """
        Implement the abstract method from BaseQAModel.
        Answers a question given a context using the specified model.
        """
        rag = RAG(model=self.model, client=self.client)
        prediction = rag.forward(context, question)
        # print(self.client.inspect_history(n=3))
        return prediction.answer
