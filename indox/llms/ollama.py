from typing import Any

from tenacity import retry, stop_after_attempt, wait_random_exponential
from loguru import logger
import sys
from indox.core import BaseLLM

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")

logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")


class Ollama(BaseLLM):
    model: str = ""

    def __init__(self, model):
        super().__init__(model=model)
        """
        Initializes the Ollama model with the specified model version and an optional prompt template.

        Args:
            model (str): Ollama model version.
        """

        try:
            logger.info(f"Initializing Ollama with model: {model}")
            self.model = model
            logger.info("Ollama initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Ollama: {e}")
            raise

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _generate_response(self, messages):
        """
        Generates a response from the Ollama model.

        Args:
            messages : The  messages to send to the model.
            max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 250.
            temperature (float, optional): The sampling temperature. Defaults to 0.

        Returns:
            str: The generated response.
        """
        import ollama as ol

        try:
            logger.info("Generating response")
            response = ol.generate(model=self.model, prompt=messages)
            result = response["response"].strip().replace("\n", "").replace("\t", "")
            logger.info("Response generated successfully")
            return result
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def _format_prompt(self, context, question):
        """
        Formats the prompt for generating a response.

        Args:
            context (str): The context for the prompt.
            question (str): The question for the prompt.

        Returns:
            str: The formatted prompt.
        """
        return f"Given Context: {context} Give the best full answer amongst the option to question {question}"

    def answer_question(self, context, question, max_tokens=350):
        """
        Public method to generate an answer to a question based on the given context.

        Args:
            context (str): The text to summarize.
            question (str): The question to answer.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 350.

        Returns:
            str: The generated answer.
        """
        try:
            logger.info("Answering question")
            prompt = self._format_prompt(context, question)
            return self._generate_response(messages=prompt, max_tokens=max_tokens, temperature=0)
        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
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
            logger.info("Generating summary for documentation")
            prompt = f"You are a helpful assistant. Give a detailed summary of the documentation provided.\n\nDocumentation:\n{documentation}"
            messages = prompt
            return self._generate_response(messages, max_tokens=150, temperature=0)
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return str(e)

    def grade_docs(self, context, question):
        """
        Answers a question using an agent-based approach with access to tools.

        Args:
            context (str): The context in which the question is asked.
            question (str): The question to answer.
        """
        filtered_docs = []
        system_prompt = """
            You are a grader assessing the relevance of a retrieved document to a user question.
            If the document contains any keywords or evidence related to the user question, even if minimal, 
            grade it as relevant. The goal is to filter out only completely erroneous retrievals.
            Give a binary "yes" or "no" score to indicate whether the document is relevant to the question.

            Provide the score with no preamble or explanation.
                        
        """
        for doc in context:
            prompt = f"Here is the retrieved document:\n{doc}\nHere is the user question:\n{question}"
            messages = system_prompt + prompt
            try:
                grade = self._generate_response(messages, max_tokens=150, temperature=0).lower()
                if grade == "yes":
                    logger.info("Relevant doc")
                    filtered_docs.append(doc)
                elif grade == "no":
                    logger.info("Not relevant doc")
            except Exception as e:
                logger.error(f"Error grading document: {e}")
        return filtered_docs

    def check_hallucination(self, context, answer):
        """
        Checks if an answer is grounded in the provided context.

        Args:
            context (str): The context for checking.
            answer (str): The answer to check.

        Returns:
            str: 'yes' if the answer is grounded, 'no' otherwise.
        """
        system_prompt = """
            You are a grader assessing whether an answer is grounded in / supported by a set of facts.
            Give a binary score 'yes' or 'no' to indicate whether the answer is grounded / supported by the set of facts.
            Provide the score with no preamble or explanation.
        """
        prompt = f"Here are the facts:\n{context}\nHere is the answer:\n{answer}"
        messages = system_prompt + prompt
        try:
            logger.info("Checking hallucination for answer")
            return self._generate_response(messages, max_tokens=150, temperature=0).lower()
        except Exception as e:
            logger.error(f"Error checking hallucination: {e}")
            return str(e)

    def chat(self, prompt):
        return self._generate_response(messages=prompt)
