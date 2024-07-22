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
    def _generate_response(self, messages, max_tokens=250, temperature=0):
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
                max_tokens=max_tokens,
            )
            result = response.choices[0].message.content.strip()
            logger.info("Response generated successfully")
            return result
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def generate_evaluation_response(self, prompt):
            """
            Generates a response to a custom prompt.

            Args:
                prompt (str): The custom prompt to generate a response for.

            Returns:
                str: The generated response.
            """
            try:
                logger.info("Generating response to custom prompt")
                messages = [
                    {"role": "system", "content": "You are a assistant for llm evaluation"},
                    {"role": "user", "content": prompt},
                ]
                return self._generate_response(messages, max_tokens=150, temperature=0)
            except Exception as e:
                logger.error(f"Error generating response to custom prompt: {e}")
                return str(e)