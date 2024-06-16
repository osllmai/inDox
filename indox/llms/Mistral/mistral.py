import logging
from tenacity import retry, stop_after_attempt, wait_random_exponential
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os

logging.basicConfig(filename='indox.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')


class Mistral:
    def __init__(self, api_key, model="mistral-medium-latest"):
        """
        Initializes the Mistral AI model with the specified model version and an optional prompt template.

        Args:
            api_key (str): The API key for Mistral AI.
            model (str): The Mistral AI model version.
        """
        try:
            logging.info("Initializing MistralAI with model: %s", model)
            self.model = model
            self.client = MistralClient(api_key=api_key)
            logging.info("MistralAI initialized successfully")
        except Exception as e:
            logging.error("Error initializing MistralAI: %s", e)
            raise

    def run_mistral(self, user_message):
        """
        Runs the Mistral model to generate a response based on the user message.

        Args:
            user_message (str): The message to be processed by the Mistral model.

        Returns:
            str: The generated response.
        """
        try:
            messages = [
                ChatMessage(role="user", content=user_message)
            ]
            chat_response = self.client.chat(
                model=self.model,
                messages=messages
            )
            return chat_response.choices[0].message.content
        except Exception as e:
            logging.error("Error in run_mistral: %s", e)
            return str(e)

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(self, context, question, max_tokens=150, stop_sequence=None, temperature=0):
        """
        Generates an answer to the given question using the Mistral AI model.

        Args:
            context (str): The text to summarize.
            question (str): The question to answer.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.
            temperature (float, optional): The sampling temperature. Defaults to 0.

        Returns:
            str: The generated summary.
        """
        prompt = f"""
        Context information is below.
        ---------------------
        {context}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        Query: {question}
        Answer:
        """

        try:
            logging.info("Attempting to generate an answer for the question: %s", question)
            return self.run_mistral(prompt)
        except Exception as e:
            logging.error("Error generating answer: %s", e)
            raise

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        """
        Public method to generate an answer to a question based on the given context.

        Args:
            context (str): The text to summarize.
            question (str): The question to answer.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        try:
            logging.info("Answering question: %s", question)
            return self._attempt_answer_question(
                context,
                question,
                max_tokens=max_tokens,
                stop_sequence=stop_sequence,
                temperature=0,
            )
        except Exception as e:
            logging.error("Error in answer_question: %s", e)
            return str(e)

    def get_summary(self, documentation):
        """
        Generates a detailed summary of the provided documentation.

        Args:
            documentation (str): The documentation to summarize.

        Returns:
            str: The generated summary.
        """
        try:
            logging.info("Generating summary for documentation")
            prompt = f"You are a helpful assistant. Give a detailed summary of the documentation provided.\n\nDocumentation:\n{documentation}"
            return self.run_mistral(prompt)
        except Exception as e:
            logging.error("Error generating summary: %s", e)
            return str(e)

    def grade_docs(self, context, question):
        """
        Answers a question using an agent-based approach with access to tools.

        Args:
            context (str): The context in which the question is asked.
            question (str): The question to answer.
        """
        filtered_docs = []
        try:
            system_prompt = f"""
            You are a grader assessing relevance of a retrieved
            document to a user question. If the document contains keywords related to the
            user question, grade it as relevant. It does not need to be a stringent test.
            The goal is to filter out erroneous retrievals.

            Give a binary score 'yes' or 'no' score to indicate whether the document is
            relevant to the question.

            Provide the score with no preamble or explanation.
            """
            for i in range(len(context)):
                prompt = f"""
                    Here is the retrieved document:
                    {context[i]}
                    Here is the user question:
                    {question}"""
                grade = self.run_mistral(system_prompt + prompt).strip()
                if grade.lower() == "yes":
                    print("Relevant doc")
                    filtered_docs.append(context[i])
                elif grade.lower() == "no":
                    print("Not Relevant doc")
            return filtered_docs
        except Exception as e:
            logging.error("Error generating agent answer: %s", e)
            return str(e)

    def check_hallucination(self, context, answer):
        try:
            system_prompt = """
                You are a grader assessing whether an answer is grounded in / supported by a set of facts.
                Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded / supported by a set
                 of facts. Provide score with no preamble or explanation.
                """
            prompt = f"""
                Here are the facts:
                \n -------- \n
                {context}
                \n -------- \n
                Here is the answer : {answer}
                """
            return self.run_mistral(system_prompt + prompt).strip()
        except Exception as e:
            logging.error("Error generating agent answer: %s", e)
            return str(e)
