import requests
from loguru import logger
import sys

# Set up logging
logger.remove()  # Remove the default logger
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)
logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


class NerdToken:
    def __init__(self, api_key: str, model: str = "openai/gpt-4o-mini"):
        """
        Initializes the NerdTokenApi with the API key and model.

        Args:
            api_key (str): The API key for accessing the Indox API.
            model (str, optional): The model to use. Defaults to "openai/gpt-4o-mini".
        """
        self.api_key = api_key
        self.model = model

    def _send_request(self, system_prompt, user_prompt, **kwargs):
        url = "https://api-token.nerdstudio.ai/v1/api/text_generation/generate/"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Default parameters
        data = {
            "frequency_penalty": kwargs.get("frequency_penalty", 0),
            "max_tokens": kwargs.get("max_tokens", 4000),
            "messages": [
                {"content": system_prompt, "role": "system"},
                {"content": user_prompt, "role": "user"},
            ],
            "model": self.model,
            "presence_penalty": kwargs.get("presence_penalty", 0),
            "stream": kwargs.get("stream", False),
            "temperature": kwargs.get("temperature", 0.3),
            "top_p": kwargs.get("top_p", 1),
        }

        try:
            response = requests.post(url, headers=headers, json=data)

            if response.status_code == 200:
                answer_data = response.json()
                generated_text = answer_data["choices"][0]["message"]["content"]
                return generated_text
            else:
                error_message = f"Error from Nerd Token API: {response.status_code}, {response.text}"
                logger.error(error_message)
                raise Exception(error_message)
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise

    def answer_question(
        self,
        context: str,
        question: str,
        prompt_template: str = "Context: {context}\nQuestion: {question}\nAnswer:",
        **kwargs,
    ):
        """
        Answer a question based on the given context using the Indox API.

        Args:
            context (str): The text to base the answer on.
            question (str): The question to be answered.
            prompt_template (str, optional): Template for formatting the prompt.
            **kwargs: Additional parameters for API call (temperature, max_tokens, etc.)

        Returns:
            str: The generated answer.
        """
        try:
            system_prompt = """
You are a highly knowledgeable assistant. You must only provide answers based on the provided context. If the context does not contain the necessary information to answer the query, respond with: "The context does not provide sufficient information to answer this query."
Do not use external knowledge, make assumptions, or fabricate information. Always base your responses entirely on the context provided.
"""
            user_prompt = prompt_template.format(
                context=context,
                question=question,
            )
            return self._send_request(system_prompt, user_prompt, **kwargs)
        except Exception as e:
            logger.error(e)
            return str(e)

    def get_summary(self, documentation: str, **kwargs):
        """
        Generates a detailed summary of the provided documentation.

        Args:
            documentation (str): The documentation to summarize.
            **kwargs: Additional parameters for API call (temperature, max_tokens, etc.)

        Returns:
            str: The generated summary.
        """
        try:
            system_prompt = "You are a helpful assistant."
            user_prompt = f"You are a helpful assistant. Give a detailed summary of the documentation provided.\n\nDocumentation:\n {documentation}"
            return self._send_request(system_prompt, user_prompt, **kwargs)
        except Exception as e:
            logger.error(e)
            return str(e)

    def grade_docs(self, context: list, question: str, **kwargs):
        """
        Grades the relevance of documents to a question.

        Args:
            context (list): List of documents to grade.
            question (str): The question to compare against.
            **kwargs: Additional parameters for API call (temperature, max_tokens, etc.)

        Returns:
            list: Filtered list of relevant documents.
        """
        filtered_docs = []
        try:
            system_prompt = """
            You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question, grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
            Provide the score with no preamble or explanation.
            """
            for doc in context:
                user_prompt = f"""
                Here is the retrieved document:
                {doc}
                Here is the user question:
                {question}
                """
                response = self._send_request(system_prompt, user_prompt, **kwargs)
                if response.lower() == "yes":
                    logger.info("Relevant doc")
                    filtered_docs.append(doc)
                elif response.lower() == "no":
                    logger.info("Not Relevant doc")
            return filtered_docs
        except Exception as e:
            logger.error(f"Error generating agent answer: {e}")
            return str(e)

    def check_hallucination(self, context: str, answer: str, **kwargs):
        """
        Checks if an answer is grounded in the provided context.

        Args:
            context (str): The text to base the answer on.
            answer (str): The answer to check for hallucination.
            **kwargs: Additional parameters for API call (temperature, max_tokens, etc.)

        Returns:
            str: 'yes' if the answer is grounded, 'no' otherwise.
        """
        try:
            system_prompt = """
            You are a grader assessing whether an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded / supported by a set of facts. Provide score with no preamble or explanation.
            """
            user_prompt = f"""
            Here are the facts:
            \n -------- \n
            {context}
            \n -------- \n
            Here is the answer: {answer}
            """
            return self._send_request(system_prompt, user_prompt, **kwargs)
        except Exception as e:
            logger.error(f"Error generating agent answer: {e}")
            return str(e)

    def chat(
        self, prompt: str, system_prompt: str = "You are a helpful assistant", **kwargs
    ):
        """
        Simple chat interface with the model.

        Args:
            prompt (str): The user's input prompt.
            system_prompt (str, optional): System prompt to set the context.
            **kwargs: Additional parameters for API call (temperature, max_tokens, etc.)

        Returns:
            str: The model's response.
        """
        return self._send_request(system_prompt, prompt, **kwargs)
