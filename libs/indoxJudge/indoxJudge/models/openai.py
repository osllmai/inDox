from tenacity import retry, stop_after_attempt, wait_random_exponential
from loguru import logger
import sys

# Set up logging
logger.remove()  # Remove the default logger
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)

logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


class OpenAi:
    """
    A class to interface with OpenAI's models for evaluation purposes.

    This class uses the OpenAI API to send requests and receive responses, which are utilized
    for evaluating the performance of language models.
    """

    def __init__(
        self, api_key: str, model: str, max_tokens: int = 2048, temperature: float = 0.2
    ):
        """
        Initializes the OpenAi class with the specified API key, model version, and max tokens.

        Args:
            api_key (str): The API key for accessing the OpenAI API.
            model (str): The GPT model version to use.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 2048.
        """
        from openai import OpenAI

        try:
            logger.info(
                f"Initializing OpenAi with model: {model} and max_tokens: {max_tokens}"
            )
            self.model = model
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            logger.error(f"Error initializing OpenAi: {e}")
            raise

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _generate_response(self, messages: list) -> str:
        """
        Generates a response from the OpenAI model.

        Args:
            messages (list): The list of messages to send to the model, formatted as a conversation.
            temperature (float, optional): The sampling temperature, influencing response randomness. Defaults to 0.00001.

        Returns:
            str: The generated response.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            result = response.choices[0].message.content.strip()
            return result
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def generate_evaluation_response(self, prompt: str) -> str:
        """
        Generates a response to a custom evaluation prompt using the OpenAI model.

        This method formats the prompt within a system and user message structure,
        with the system message indicating that the assistant is for LLM evaluation.

        Args:
            prompt (str): The custom prompt to generate a response for.

        Returns:
            str: The generated response.
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an assistant for LLM evaluation",
                },
                {"role": "user", "content": prompt},
            ]
            response = self._generate_response(messages)
            if response.startswith("```json") and response.endswith("```"):
                response = response[7:-3].strip()
            return response
        except Exception as e:
            logger.error(f"Error generating response to custom prompt: {e}")
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

        messages = [
            {
                "role": "system",
                "content": "your are a helpful assistant to analyze charts",
            },
            {"role": "user", "content": prompt},
        ]
        response = self._generate_response(messages=messages)

        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        return response
