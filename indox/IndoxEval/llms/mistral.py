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
    def __init__(self, api_key, model="mistral-medium-latest"):
        """
        Initializes the Mistral AI model with the specified model version and an optional prompt template.

        Args:
            api_key (str): The API key for Mistral AI.
            model (str): The Mistral AI model version.
        """
        from mistralai.client import MistralClient
        try:
            logger.info(f"Initializing MistralAI with model: {model}")
            self.model = model
            self.client = MistralClient(api_key=api_key)
            logger.info("MistralAI initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MistralAI: {e}")
            raise

    def _run_mistral(self, user_message):
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
            return chat_response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in run_mistral: {e}")
            return str(e)

    def generate_evaluation_response(self, prompt):

        try:
            system_prompt = "You are a assistant for llm evaluation"

            response = self._run_mistral(system_prompt + prompt).strip()
            return response
        except Exception as e:
            logger.error(f"Error generating agent answer: {e}")
            return str(e)
