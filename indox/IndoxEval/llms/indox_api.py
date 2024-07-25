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


class IndoxApi:
    """
    A class to interface with the Indox API for generating responses for evaluation purposes.

    This class uses the Indox API to send requests and receive responses, which are utilized
    for evaluating the performance of language models.
    """

    def __init__(self, api_key: str, prompt_template: str = None):
        """
        Initializes the IndoxApi with the specified API key and an optional prompt template.

        Args:
            api_key (str): The API key for accessing the Indox API.
            prompt_template (str, optional): The template for formatting prompts. Defaults to None.
        """
        self.api_key = api_key
        self.prompt_template = prompt_template or "Context: {context}\nQuestion: {question}\nAnswer:"

    def _send_request(self, system_prompt: str, user_prompt: str) -> str:
        """
        Sends a request to the Indox API to generate a response.

        Args:
            system_prompt (str): The system prompt to include in the request.
            user_prompt (str): The user prompt to generate a response for.

        Returns:
            str: The generated response text.

        Raises:
            Exception: If there is an error during the API request.
        """
        url = 'http://5.78.55.161/api/chat_completion/generate/'
        headers = {
            'accept': '*/*',
            "Authorization": f"Bearer {self.api_key}",
            'Content-Type': 'application/json',
        }

        data = {
            "frequency_penalty": 0,
            "max_tokens": 1000,
            "messages": [
                {
                    "content": system_prompt,
                    "role": "system"
                },
                {
                    "content": user_prompt,
                    "role": "user"
                }
            ],
            "model": "gpt-3.5-turbo-0125",
            "presence_penalty": 0,
            "stream": True,
            "temperature": 0.3,
            "top_p": 1
        }

        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            answer_data = response.json()
            generated_text = answer_data.get("text_message", "")
            return generated_text
        else:
            error_message = f"Error from Indox API: {response.status_code}, {response.text}"
            logger.error(error_message)
            raise Exception(error_message)

    def generate_evaluation_response(self, prompt: str) -> str:
        """
        Generates an evaluation response using the Indox API.

        This method adds a system prompt indicating that the response is for evaluation purposes,
        and then sends the prompt to the API for generating a response.

        Args:
            prompt (str): The prompt to evaluate.

        Returns:
            str: The generated evaluation response.
        """
        try:
            logger.info("Generating evaluation response")
            system_prompt = "You are an assistant for LLM evaluation."

            response = self._send_request(system_prompt, prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating evaluation response: {e}")
            return str(e)
