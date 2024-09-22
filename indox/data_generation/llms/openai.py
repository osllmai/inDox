import requests
from functools import lru_cache
from typing import Iterable, Generator
import tiktoken
class OpenAI:
    def __init__(self, api_key, model_name, **kwargs):
        self.api_key = api_key
        self.model_name = model_name
        self.chat_prompt_template = kwargs.get("chat_prompt_template", None)
        self.system_prompt = kwargs.get("system_prompt", None)

    @lru_cache(maxsize=None)
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the input text."""
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    @lru_cache(maxsize=None)
    def get_max_context_length(self, max_new_tokens: int = 0) -> int:
        """Returns the maximum context length allowed by the model."""
        return 7000 - max_new_tokens

    def run(self, prompts: Iterable[str], max_new_tokens: int = 0, **kwargs) -> Generator[str, None, None]:
        prompts_tuple = tuple(prompts)

        for prompt in prompts_tuple:
            system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")
            yield self.generate_response(system_prompt, prompt, **kwargs)

    def generate_response(self, system_prompt, user_prompt, **kwargs):
        return self._send_request(system_prompt, user_prompt, **kwargs)

    def _send_request(self, system_prompt, user_prompt, **kwargs):
        url = 'http://5.78.55.161/api/chat_completion/generate/'
        headers = {
            'accept': '*/*',
            "Authorization": f"Bearer {self.api_key}",
            'Content-Type': 'application/json',
        }

        data = {
            "frequency_penalty": 0,
            "max_tokens": kwargs.get("max_new_tokens", 150),
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
            "model": f"{self.model_name}",
            "presence_penalty": 0,
            "stream": True,
            "temperature": kwargs.get("temperature", 0.3),
            "top_p": kwargs.get("top_p", 1)
        }

        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            answer_data = response.json()
            return answer_data.get("text_message", "")
        else:
            raise Exception(f"Error From Indox API: {response.status_code}, {response.text}")
