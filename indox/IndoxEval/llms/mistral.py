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


class Mistral:
    """
    A class to interface with the Mistral AI model for evaluation purposes.

    This class uses the Mistral AI client to send requests and receive responses,
    which are utilized for evaluating the performance of language models.
    """
    def __init__(self, api_key: str, model: str = "mistral-medium-latest"):
        """
        Initializes the Mistral class with the specified API key and model version.

        Args:
            api_key (str): The API key for accessing the Mistral AI.
            model (str): The Mistral AI model version to use. Defaults to "mistral-medium-latest".
        """
        from mistralai.client import MistralClient
        try:
            logger.info(f"Initializing Mistral with model: {model}")
            self.model = model
            self.client = MistralClient(api_key=api_key)
            logger.info("Mistral initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Mistral: {e}")
            raise

    def _run_mistral(self, user_message: str) -> str:
        """
        Runs the Mistral model to generate a response based on the user message.

        Args:
            user_message (str): The message to be processed by the Mistral model.

        Returns:
            str: The generated response.
        """
        from mistralai.models.chat_completion import ChatMessage

        try:
            messages = [
                ChatMessage(role="user", content=user_message)
            ]
            chat_response = self.client.chat(
                model=self.model,
                messages=messages
            )
            return chat_response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in _run_mistral: {e}")
            return str(e)

    def generate_evaluation_response(self, prompt: str) -> str:
        """
        Generates an evaluation response using the Mistral AI model.

        This method adds a system prompt indicating that the response is for evaluation purposes,
        and then processes the user prompt to generate a response.

        Args:
            prompt (str): The prompt to evaluate.

        Returns:
            str: The generated evaluation response.
        """
        try:
            logger.info("Generating evaluation response")
            system_prompt = "You are an assistant for LLM evaluation."

            response = self._run_mistral(system_prompt + prompt).strip()
            return response
        except Exception as e:
            logger.error(f"Error generating evaluation response: {e}")
            return str(e)
