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
    def __init__(self, api_key, model="gemini-1.5-flash-latest"):
        """
        Initializes with the specified model version and an optional prompt template.

        Args:
            api_key (str): The API key for Google Ai.
            model (str): The Gemini model version.
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
    def _generate_response(self, prompt):
        """
        Generates a response using the model.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response text.
        """
        try:
            logger.info("Generating response")
            response = self.model.generate_content(contents=prompt)
            logger.info("Response in generated successfully")
            return response.text.strip().replace("\n", "")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def generate_evaluation_response(self, prompt):
        try:
            logger.info("Answering question")
            system_prompt = "You are a assistant for llm evaluation"
            return self._generate_response(system_prompt + prompt)
        except Exception as e:
            logger.error(f"Error : {e}")
            return str(e)
