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


class Mistral:
    """
    A class to interface with the Mistral AI model for evaluation purposes.

    This class uses the Mistral AI client to send requests and receive responses,
    which are utilized for evaluating the performance of language models.
    """

    def __init__(self, api_key: str, model: str = "mistral-medium-latest"):
        """
        Initializes the Mistral class with the specified API key and model version.

        Args:
            api_key (str): The API key for accessing the Mistral AI.
            model (str): The Mistral AI model version to use. Defaults to "mistral-medium-latest".
        """
        from mistralai.client import MistralClient

        try:
            logger.info(f"Initializing Mistral with model: {model}")
            self.model = model
            self.client = MistralClient(api_key=api_key)
            logger.info("Mistral initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Mistral: {e}")
            raise

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _run_mistral(self, user_message: str) -> str:
        """
        Runs the Mistral model to generate a response based on the user message.
        Implements exponential backoff retry mechanism for API calls.

        Args:
            user_message (str): The message to be processed by the Mistral model.

        Returns:
            str: The generated response.

        Retries:
            - Up to 6 attempts (1 initial + 5 retries)
            - Exponential backoff with randomization (1-20 seconds)
        """
        from mistralai.models.chat_completion import ChatMessage

        try:
            messages = [ChatMessage(role="user", content=user_message)]
            chat_response = self.client.chat(model=self.model, messages=messages)
            return chat_response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in _run_mistral: {e}")
            raise  # Raising the exception to trigger retry

    def generate_evaluation_response(self, prompt: str) -> str:
        """
        Generates an evaluation response using the Mistral AI model.

        This method adds a system prompt indicating that the response is for evaluation purposes,
        and then processes the user prompt to generate a response.

        Args:
            prompt (str): The prompt to evaluate.

        Returns:
            str: The generated evaluation response.
        """
        try:
            system_prompt = "You are an assistant for LLM evaluation."

            response = self._run_mistral(system_prompt + prompt).strip()
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

        system_prompt = "your are a helpful assistant to analyze charts."
        messages = system_prompt + "\n" + prompt
        response = self._run_mistral(messages)

        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        return response
