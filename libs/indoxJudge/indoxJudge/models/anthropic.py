from loguru import logger
import sys
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Set up logging
logger.remove()  # Remove the default logger
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)

logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


class Anthropic:
    """
    A class to interface with Anthropic's Claude model for evaluation purposes.

    This class uses the Anthropic API to generate responses from Claude,
    which can be used for evaluating language model outputs.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-opus-20240229",
        prompt_template: str = None,
    ):
        """
        Initializes the AnthropicModel with the specified model and prompt template.

        Args:
            api_key (str): The API key for accessing the Anthropic API.
            model (str, optional): The model version to use. Defaults to "claude-3-opus-20240229".
            prompt_template (str, optional): The template for formatting prompts. Defaults to None.
        """
        from anthropic import Anthropic

        try:
            logger.info(f"Initializing AnthropicModel with model: {model}")
            self.model = model
            self.client = Anthropic(api_key=api_key)
            self.prompt_template = (
                prompt_template or "Context: {context}\nQuestion: {question}\nAnswer:"
            )
            if not api_key:
                raise ValueError("A valid Anthropic API key is required.")
            logger.info("AnthropicModel initialized successfully")
        except ValueError as ve:
            logger.error(f"ValueError during initialization: {ve}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during initialization: {e}")
            raise

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _send_request(self, prompt: str) -> str:
        """
        Sends a request to the Anthropic API to generate a response.
        Implements exponential backoff retry mechanism for API calls.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The generated response text.

        Raises:
            Exception: If there is an error during the API request after all retry attempts.

        Retries:
            - Up to 6 attempts (1 initial + 5 retries)
            - Exponential backoff with randomization (1-20 seconds)
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error in _send_request: {e}")
            raise  # This will trigger a retry

    def generate_evaluation_response(self, prompt: str) -> str:
        """
        Generates an evaluation response using the Anthropic model.

        This method adds a system prompt indicating that the response is for evaluation purposes,
        and then sends the prompt to the model for generating a response.

        Args:
            prompt (str): The prompt to evaluate.

        Returns:
            str: The generated evaluation response.
        """
        try:
            system_prompt = """
            You are a grader assessing for LLM evaluation.
            """

            response = self._send_request(system_prompt + "\n" + prompt)
            if response.startswith("```json") and response.endswith("```"):
                response = response[7:-3].strip()
            return response
        except Exception as e:
            logger.error(f"Error generating evaluation response: {e}")
            return str(e)

    def generate_interpretation(self, models_data, mode):
        """
        Generates interpretation based on the given mode and models data.

        Args:
            models_data: Data about the models to be interpreted
            mode (str): The mode of interpretation ('comparison', 'rag', 'safety', or 'llm')

        Returns:
            str: The generated interpretation
        """
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

        system_prompt = "you are a helpful assistant to analyze charts."
        messages = system_prompt + "\n" + prompt
        response = self._send_request(messages)

        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        return response
