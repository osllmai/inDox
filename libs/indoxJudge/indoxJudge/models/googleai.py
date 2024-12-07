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


class GoogleAi:
    """
    A class to interface with Google AI's generative models for evaluation purposes.

    This class is specifically designed to use a Google AI model to generate responses
    for the purpose of evaluating language model outputs.
    """

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash-latest"):
        """
        Initializes the GoogleAi class with the specified model version.

        Args:
            api_key (str): The API key for accessing Google AI services.
            model (str): The model version to use for generating responses.
        """
        import google.generativeai as genai
        try:
            logger.info(f"Initializing GoogleAi with model: {model}")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
            logger.info("GoogleAi initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing GoogleAi: {e}")
            raise

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _generate_response(self, prompt: str) -> str:
        """
        Generates a response using the Google AI model.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response text.
        """
        try:
            response = self.model.generate_content(contents=prompt)
            return response.text.strip().replace("\n", "")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def generate_evaluation_response(self, prompt: str) -> str:
        """
        Generates an evaluation response using the Google AI model.

        This method prepends a system prompt indicating that the response is for evaluation purposes,
        and then generates the response.

        Args:
            prompt (str): The prompt to evaluate.

        Returns:
            str: The generated evaluation response.
        """
        try:
            logger.info("Generating evaluation response")
            system_prompt = "You are an assistant for LLM evaluation."
            response = self._generate_response(system_prompt + prompt)
            if response.startswith("```json") and response.endswith("```"):
                response = response[7:-3].strip()
            return response
        except Exception as e:
            logger.error(f"Error generating evaluation response: {e}")
            return str(e)

    def generate_interpretation(self, models_data, mode):
        prompt = ""
        if mode == "comparison":
            from .interpretation_template.comparison_template import ModelComparisonTemplate
            prompt = ModelComparisonTemplate.generate_comparison(models=models_data, mode="llm model quality")
        elif mode == "rag":
            from .interpretation_template.rag_interpretation_template import RAGEvaluationTemplate
            prompt = RAGEvaluationTemplate.generate_interpret(data=models_data)
        elif mode == "safety":
            from .interpretation_template.safety_interpretation_template import SafetyEvaluationTemplate
            prompt = SafetyEvaluationTemplate.generate_interpret(data=models_data)
        elif mode == "llm":
            from .interpretation_template.llm_interpretation_template import LLMEvaluatorTemplate
            prompt = LLMEvaluatorTemplate.generate_interpret(data=models_data)

        system_prompt = "your are a helpful assistant to analyze charts."
        messages = system_prompt + "\n" + prompt
        response = self._generate_response(messages=messages)

        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        return response
