import requests
from loguru import logger
import sys
from indoxArcg.core import BaseLLM

# Set up logging
logger.remove()  # Remove the default logger
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)

logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


class NerdToken:
    def __init__(
        self,
        api_key: str,
        max_tokens: int = 4000,
        temperature: float = 0.3,
        stream: bool = False,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        top_p: float = 1,
        prompt_template: str = None,
    ):
        """
        Initializes the NerdTokenApi with the specified API key, model, and an optional prompt template.

        Args:
            api_key (str): The API key for accessing the Indox API.
            max_tokens (int, optional): The maximum number of tokens for the response. Defaults to 4000.
            temperature (float, optional): Sampling temperature. Defaults to 0.3.
            stream (bool, optional): Whether to stream responses. Defaults to False.
            presence_penalty (float, optional): Presence penalty for text generation. Defaults to 0.
            frequency_penalty (float, optional): Frequency penalty for text generation. Defaults to 0.
            top_p (float, optional): Nucleus sampling parameter. Defaults to 1.
            prompt_template (str, optional): The template for formatting prompts. Defaults to None.
        """
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stream = stream
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.top_p = top_p
        self.prompt_template = (
            prompt_template or "Context: {context}\nQuestion: {question}\nAnswer:"
        )

    def _send_request(self, system_prompt, user_prompt):
        url = "https://api-token.nerdstudio.ai/v1/api/text_generation/generate/"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "frequency_penalty": self.frequency_penalty,
            "max_tokens": self.max_tokens,
            "messages": [
                {"content": system_prompt, "role": "system"},
                {"content": user_prompt, "role": "user"},
            ],
            "model": "openai/gpt-4o-mini",
            "presence_penalty": self.presence_penalty,
            "stream": self.stream,
            "temperature": self.temperature,
            "top_p": self.top_p,
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

    def _attempt_answer_question(
        self,
        context,
        question,
    ):
        """
        Generates an answer to a question based on the given context using the Indox API.

        Args:
            context (str): The text to base the answer on.
            question (str): The question to be answered.

        Returns:
            str: The generated answer.
        """

        system_prompt = """
 You are a highly knowledgeable assistant. You must only provide answers based on the provided context. If the context does not contain the necessary information to answer the query, respond with: "The context does not provide sufficient information to answer this query.
 Do not use external knowledge, make assumptions, or fabricate information. Always base your responses entirely on the context provided.
"""

        user_prompt = self.prompt_template.format(
            context=context,
            question=question,
        )
        return self._send_request(
            system_prompt,
            user_prompt,
        )

    def answer_question(
        self,
        context,
        question,
    ):
        """
        Answer a question based on the given context using the Indox API.

        Returns:
            str: The generated answer.
        """
        try:
            # prompt_template = prompt_template or self.prompt_template
            return self._attempt_answer_question(
                context,
                question,
            )
        except Exception as e:
            logger.error(e)
            return str(e)

    def get_summary(
        self,
        documentation,
    ):
        """
        Generates a detailed summary of the provided documentation.


        Returns:
            str: The generated summary.
        """
        try:
            system_prompt = "You are a helpful assistant."
            user_prompt = f"You are a helpful assistant. Give a detailed summary of the documentation provided.\n\nDocumentation:\n {documentation}"
            return self._send_request(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except Exception as e:
            logger.error(e)
            return str(e)

    def grade_docs(
        self,
        context,
        question,
    ):
        """
        Answers a question using an agent-based approach with access to tools.

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
            for i in range(len(context)):
                user_prompt = f"""
                Here is the retrieved document:
                {context[i]}
                Here is the user question:
                {question}
                """
                response = self._send_request(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                if response.lower() == "yes":
                    logger.info("Relevant doc")
                    filtered_docs.append(context[i])
                elif response.lower() == "no":
                    logger.info("Not Relevant doc")
            return filtered_docs
        except Exception as e:
            logger.error(f"Error generating agent answer: {e}")
            return str(e)

    def check_hallucination(
        self,
        context,
        answer,
    ):
        """
        Checks if an answer is grounded in the provided context.

        Args:
            context (str): The text to base the answer on.
            answer (str): The answer to check for hallucination.

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
            response = self._send_request(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            return response
        except Exception as e:
            logger.error(f"Error generating agent answer: {e}")
            return str(e)

    # def _assistant(self, system_prompt, user_prompt, role, history):
    #     try:
    #         response = self._send_request(system_prompt=system_prompt, user_prompt=user_prompt, role=role,
    #                                       history=history)
    #         return response
    #     except Exception as e:
    #         logger.error(f"Error generating agent answer: {e}")
    #         return str(e)

    def chat(
        self,
        prompt,
        system_prompt="You are a helpful assistant",
    ):
        return self._send_request(
            system_prompt=system_prompt,
            user_prompt=prompt,
        )
