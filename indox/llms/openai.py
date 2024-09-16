from tenacity import retry, stop_after_attempt, wait_random_exponential
from loguru import logger
import sys
from indox.core import BaseLLM
from pydantic import ConfigDict

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")

logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")


class OpenAi:
    def __init__(self, api_key, model):
        """
        Initializes the GPT-3 model with the specified model version and an optional prompt template.

        Args:
            api_key (str): The API key for OpenAI.
            model (str): The GPT-3 model version.
        """
        from openai import OpenAI

        try:
            logger.info(f"Initializing OpenAi with model: {model}")
            self.model = model
            self.client = OpenAI(api_key=api_key)
            logger.info("OpenAi initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing OpenAi: {e}")
            raise

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _generate_response(self, messages, max_tokens, temperature):
        """
        Generates a response from the OpenAI model.

        Args:
            messages (list): The list of messages to send to the model.
            max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 250.
            temperature (float, optional): The sampling temperature. Defaults to 0.

        Returns:
            str: The generated response.
        """
        try:
            logger.info("Generating response")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            result = response.choices[0].message.content.strip()
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

    def answer_question(self, context, question, max_tokens=350, temperature=0.3):
        """
        Public method to generate an answer to a question based on the given context.

        Args:
            context (str): The text to summarize.
            question (str): The question to answer.
            max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 350.
            temperature (float, optional): The temperature of the generated response. Defaults to 0.3.

        Returns:
            str: The generated answer.
        """
        try:
            logger.info("Answering question")
            prompt = self._format_prompt(context, question)
            messages = [
                {"role": "system", "content": "You are Question Answering Portal"},
                {"role": "user", "content": prompt},
            ]
            return self._generate_response(messages, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            return str(e)

    def get_summary(self, documentation, max_tokens=350, temperature=0.3):
        """
        Generates a detailed summary of the provided documentation.

        Args:
            documentation (str): The documentation to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 350.
            temperature (float, optional): The temperature of the generated response. Defaults to 0.3.
        Returns:
            str: The generated summary.
        """
        try:
            logger.info("Generating summary for documentation")
            prompt = f"You are a helpful assistant. Give a detailed summary of the documentation provided.\n\nDocumentation:\n{documentation}"
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ]
            return self._generate_response(messages, max_tokens=max_tokens, temperature=temperature)
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
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            try:
                grade = self._generate_response(messages, max_tokens=150, temperature=0.0000001).lower()
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
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            logger.info("Checking hallucination for answer")
            return self._generate_response(messages, max_tokens=150, temperature=0.00001).lower()
        except Exception as e:
            logger.error(f"Error checking hallucination: {e}")
            return str(e)

    def chat(self, prompt, max_tokens=250, temperature=0.00001):
        messages = [
            {"role": "system", "content": "You are Question Answering Portal"},
            {"role": "user", "content": prompt},
        ]
        return self._generate_response(messages=messages, max_tokens=max_tokens, temperature=temperature)
