import requests
from loguru import logger
import sys
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Set up logging
logger.remove()
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)

logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


class NerdTokenApi:
    def __init__(
        self,
        api_key: str,
        max_tokens: int = 4000,
        temperature: float = 0.3,
        stream: bool = False,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        top_p: float = 1,
        prompt_template: str = None,
    ):
        """
        Initializes the NerdTokenApi with the specified API key, model, and an optional prompt template.

        Args:
            api_key (str): The API key for accessing the Indox API.
            max_tokens (int, optional): The maximum number of tokens for the response. Defaults to 4000.
            temperature (float, optional): Sampling temperature. Defaults to 0.3.
            stream (bool, optional): Whether to stream responses. Defaults to False.
            presence_penalty (float, optional): Presence penalty for text generation. Defaults to 0.
            frequency_penalty (float, optional): Frequency penalty for text generation. Defaults to 0.
            top_p (float, optional): Nucleus sampling parameter. Defaults to 1.
            prompt_template (str, optional): The template for formatting prompts. Defaults to None.
        """
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stream = stream
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.top_p = top_p
        self.prompt_template = (
            prompt_template or "Context: {context}\nQuestion: {question}\nAnswer:"
        )

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _send_request(self, system_prompt: str, user_prompt: str) -> str:
        """
        Sends a request to the Indox API to generate a response.
        Implements exponential backoff retry mechanism for API calls.

        Args:
            system_prompt (str): The system prompt to include in the request.
            user_prompt (str): The user prompt to generate a response for.

        Returns:
            str: The generated response text.

        Raises:
            Exception: If there is an error during the API request after all retry attempts.

        Retries:
            - Up to 6 attempts (1 initial + 5 retries)
            - Exponential backoff with randomization (1-20 seconds)
        """
        url = "https://api-token.nerdstudio.ai/v1/api/text_generation/generate/"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "frequency_penalty": self.frequency_penalty,
            "max_tokens": self.max_tokens,
            "messages": [
                {"content": system_prompt, "role": "system"},
                {"content": user_prompt, "role": "user"},
            ],
            "model": "gpt-4o-mini",
            "presence_penalty": self.presence_penalty,
            "stream": self.stream,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                answer_data = response.json()
                generated_text = answer_data["choices"][0]["message"]["content"]
                return generated_text
            else:
                error_message = (
                    f"Error from Indox API: {response.status_code}, {response.text}"
                )
                logger.error(error_message)
                raise Exception(error_message)  # This will trigger a retry
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise  # This will trigger a retry

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
            system_prompt = "You are an assistant for LLM evaluation."

            response = self._send_request(system_prompt, prompt)
            if response.startswith("```json") and response.endswith("```"):
                response = response[7:-3].strip()
            return response
        except Exception as e:
            logger.error(f"Error generating evaluation response: {e}")
            return str(e)

    def generate_interpretation(self, models_data, mode):
        prompt = ""
        if mode == "comparison":
            from .interpretation_template.comparison_template import (
                ModelComparisonTemplate,
            )

            prompt = ModelComparisonTemplate.generate_comparison(
                models=models_data, mode="llm model quality"
            )
        elif mode == "rag":
            from .interpretation_template.rag_interpretation_template import (
                RAGEvaluationTemplate,
            )

            prompt = RAGEvaluationTemplate.generate_interpret(data=models_data)
        elif mode == "safety":
            from .interpretation_template.safety_interpretation_template import (
                SafetyEvaluationTemplate,
            )

            prompt = SafetyEvaluationTemplate.generate_interpret(data=models_data)
        elif mode == "llm":
            from .interpretation_template.llm_interpretation_template import (
                LLMEvaluatorTemplate,
            )

            prompt = LLMEvaluatorTemplate.generate_interpret(data=models_data)

        response = self._send_request(
            system_prompt="You are a helpful assistant to analyze charts",
            user_prompt=prompt,
        )
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        return response
