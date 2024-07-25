from tenacity import retry, stop_after_attempt, wait_random_exponential
from loguru import logger
import sys

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")

logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")


class OpenAi:
    """
    A class to interface with OpenAI's GPT-3 models for evaluation purposes.

    This class uses the OpenAI API to send requests and receive responses, which are utilized
    for evaluating the performance of language models.
    """
    def __init__(self, api_key: str, model: str):
        """
        Initializes the OpenAi class with the specified API key and model version.

        Args:
            api_key (str): The API key for accessing the OpenAI API.
            model (str): The GPT-3 model version to use.
        """
        from openai import OpenAI

        try:
            logger.info(f"Initializing OpenAi with model: {model}")
            self.model = model
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            logger.error(f"Error initializing OpenAi: {e}")
            raise

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _generate_response(self, messages: list, max_tokens: int = 250, temperature: float = 0) -> str:
        """
        Generates a response from the OpenAI model.

        Args:
            messages (list): The list of messages to send to the model, formatted as a conversation.
            max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 250.
            temperature (float, optional): The sampling temperature, influencing response randomness. Defaults to 0.

        Returns:
            str: The generated response.
        """
        try:
            logger.info("Generating response")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result = response.choices[0].message.content.strip()
            return result
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def generate_evaluation_response(self, prompt: str) -> str:
        """
        Generates a response to a custom evaluation prompt using the OpenAI model.

        This method formats the prompt within a system and user message structure,
        with the system message indicating that the assistant is for LLM evaluation.

        Args:
            prompt (str): The custom prompt to generate a response for.

        Returns:
            str: The generated response.
        """
        try:
            messages = [
                {"role": "system", "content": "You are an assistant for LLM evaluation"},
                {"role": "user", "content": prompt},
            ]
            return self._generate_response(messages, max_tokens=150, temperature=0)
        except Exception as e:
            logger.error(f"Error generating response to custom prompt: {e}")
            return str(e)
