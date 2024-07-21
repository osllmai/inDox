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
    def __init__(self, api_key, prompt_template=None):
        """
        Initializes the IndoxApiOpenAiQa with the specified API key and an optional prompt template.

        Args:
            api_key (str): The API key for Indox API.
            prompt_template (str, optional): The template for the prompt. Defaults to None.
        """
        self.api_key = api_key
        self.prompt_template = prompt_template or "Context: {context}\nQuestion: {question}\nAnswer:"

    def _send_request(self, system_prompt, user_prompt):
        url = 'http://5.78.55.161/api/chat_completion/generate/'
        headers = {
            'accept': '*/*',
            "Authorization": f"Bearer {self.api_key}",
            'Content-Type': 'application/json',
        }

        data = {
            "frequency_penalty": 0,
            "max_tokens": 150,
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
            raise Exception(f"Error From Indox API: {response.status_code}, {response.text}")

    def generate_evaluation_response(self, prompt):

        try:
            system_prompt = "You are a assistant for llm evaluation"

            response = self._send_request(system_prompt, prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating agent answer: {e}")
            return str(e)
