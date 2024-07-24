import requests
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


class HuggingFaceModel:
    def __init__(self, api_key, model="mistralai/Mistral-7B-Instruct-v0.2", prompt_template=None):
        """
        Initializes the specified model via the Hugging Face Inference API.

        Args:
            api_key (str): The API key for Hugging Face.
            model (str, optional): The model version to use. Defaults to "mistralai/Mistral-7B-Instruct-v0.2".
            prompt_template (str, optional): The template for the prompt. Defaults to None.
        """
        try:
            logger.info(f"Initializing HuggingFaceModel with model: {model}")
            self.model = model
            self.api_key = api_key
            self.prompt_template = prompt_template or "Context: {context}\nQuestion: {question}\nAnswer:"
            if not self.api_key:
                raise ValueError("A valid Hugging Face API key is required.")
            logger.info("HuggingFaceModel initialized successfully")
        except ValueError as ve:
            logger.error(f"ValueError during initialization: {ve}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during initialization: {e}")
            raise

    def _send_request(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "inputs": prompt,
        }

        try:
            logger.info("Sending request to Hugging Face API")
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{self.model}",
                headers=headers,
                json=payload,
            )

            if response.status_code == 200:
                logger.info("Received successful response from Hugging Face API")
                answer_data = response.json()
                if isinstance(answer_data, list) and len(answer_data) > 0:
                    answer_data = answer_data[0]
                generated_text = answer_data.get("generated_text", "")
                # Extract only the answer part
                return generated_text
            else:
                error_message = f"Error from Hugging Face API: {response.status_code}, {response.text}"
                logger.error(error_message)
                raise Exception(error_message)
        except Exception as e:
            logger.error(f"Error in _send_request: {e}")
            raise

    def generate_evaluation_response(self, prompt):

        try:
            system_prompt = """
            You are a grader assessing for llm evaluation
            """

            response = self._send_request(system_prompt + "\n" + prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating agent answer: {e}")
            return str(e)
