import requests
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


class HuggingFaceModel:
    """
    A class to interface with Hugging Face's generative models for evaluation purposes.

    This class uses the Hugging Face Inference API to generate responses from a specified model,
    which can be used for evaluating language model outputs.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        prompt_template: str = None,
    ):
        """
        Initializes the HuggingFaceModel with the specified model and prompt template.

        Args:
            api_key (str): The API key for accessing the Hugging Face Inference API.
            model (str, optional): The model version to use. Defaults to "mistralai/Mistral-7B-Instruct-v0.2".
            prompt_template (str, optional): The template for formatting prompts. Defaults to None.
        """
        try:
            logger.info(f"Initializing HuggingFaceModel with model: {model}")
            self.model = model
            self.api_key = api_key
            self.prompt_template = (
                prompt_template or "Context: {context}\nQuestion: {question}\nAnswer:"
            )
            if not self.api_key:
                raise ValueError("A valid Hugging Face API key is required.")
            logger.info("HuggingFaceModel initialized successfully")
        except ValueError as ve:
            logger.error(f"ValueError during initialization: {ve}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during initialization: {e}")
            raise

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _send_request(self, prompt: str) -> str:
        """
        Sends a request to the Hugging Face Inference API to generate a response.
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
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": prompt,
        }

        try:
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{self.model}",
                headers=headers,
                json=payload,
            )

            if response.status_code == 200:
                answer_data = response.json()
                if isinstance(answer_data, list) and len(answer_data) > 0:
                    answer_data = answer_data[0]
                generated_text = answer_data.get("generated_text", "")
                return generated_text
            else:
                error_message = f"Error from Hugging Face API: {response.status_code}, {response.text}"
                logger.error(error_message)
                raise Exception(error_message)  # This will trigger a retry
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise  # This will trigger a retry
        except Exception as e:
            logger.error(f"Error in _send_request: {e}")
            raise  # This will trigger a retry

    def generate_evaluation_response(self, prompt: str) -> str:
        """
        Generates an evaluation response using the Hugging Face model.

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
        response = self._send_request(messages)

        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        return response
