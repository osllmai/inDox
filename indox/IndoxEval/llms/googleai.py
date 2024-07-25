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


class GoogleAi:
    """
    A class to interface with Google AI's generative models for evaluation purposes.

    This class is specifically designed to use a Google AI model to generate responses
    for the purpose of evaluating language model outputs.
    """

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash-latest"):
        """
        Initializes the GoogleAi class with the specified model version.

        Args:
            api_key (str): The API key for accessing Google AI services.
            model (str): The model version to use for generating responses.
        """
        import google.generativeai as genai
        try:
            logger.info(f"Initializing GoogleAi with model: {model}")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
            logger.info("GoogleAi initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing GoogleAi: {e}")
            raise

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _generate_response(self, prompt: str) -> str:
        """
        Generates a response using the Google AI model.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response text.
        """
        try:
            logger.info("Generating response")
            response = self.model.generate_content(contents=prompt)
            logger.info("Response generated successfully")
            return response.text.strip().replace("\n", "")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def generate_evaluation_response(self, prompt: str) -> str:
        """
        Generates an evaluation response using the Google AI model.

        This method prepends a system prompt indicating that the response is for evaluation purposes,
        and then generates the response.

        Args:
            prompt (str): The prompt to evaluate.

        Returns:
            str: The generated evaluation response.
        """
        try:
            logger.info("Generating evaluation response")
            system_prompt = "You are an assistant for LLM evaluation."
            return self._generate_response(system_prompt + prompt)
        except Exception as e:
            logger.error(f"Error generating evaluation response: {e}")
            return str(e)
