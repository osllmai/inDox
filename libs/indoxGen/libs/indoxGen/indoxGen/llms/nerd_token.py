import requests
from loguru import logger
import sys
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Set up logging
logger.remove()
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)

logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


class NerdTokenApi:
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

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _send_request(self, system_prompt: str, user_prompt: str) -> str:
        """
        Sends a request to the Indox API to generate a response.
        Implements exponential backoff retry mechanism for API calls.

        Args:
            system_prompt (str): The system prompt to include in the request.
            user_prompt (str): The user prompt to generate a response for.

        Returns:
            str: The generated response text.

        Raises:
            Exception: If there is an error during the API request after all retry attempts.

        Retries:
            - Up to 6 attempts (1 initial + 5 retries)
            - Exponential backoff with randomization (1-20 seconds)
        """
        url = "https://api-token.nerdstudio.ai/api/v1/text_generation/generate/"

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
                error_message = (
                    f"Error from Indox API: {response.status_code}, {response.text}"
                )
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
        system_prompt = "You are a helpful assistant."
        user_prompt = self.prompt_template.format(
            context=context,
            question=question,
        )
        return self._send_request(system_prompt, user_prompt)

    def chat(
        self,
        prompt,
        system_prompt="You are a helpful assistant",
    ):
        return self._send_request(
            system_prompt=system_prompt,
            user_prompt=prompt,
        )
