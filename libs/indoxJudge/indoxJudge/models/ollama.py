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


class Ollama:
    model: str = ""

    def __init__(self, model):
        super().__init__(model=model)
        """
        Initializes the Ollama model with the specified model version and an optional prompt template.

        Args:
            model (str): Ollama model version.
        """

        try:
            logger.info(f"Initializing Ollama with model: {model}")
            self.model = model
            logger.info("Ollama initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Ollama: {e}")
            raise

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _generate_response(self, messages):
        """
        Generates a response from the Ollama model.

        Args:
            messages : The  messages to send to the model.
            max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 250.
            temperature (float, optional): The sampling temperature. Defaults to 0.

        Returns:
            str: The generated response.
        """
        import ollama as ol

        try:
            logger.info("Generating response")
            response = ol.generate(model=self.model, prompt=messages)
            result = response["response"].strip().replace("\n", "").replace("\t", "")
            logger.info("Response generated successfully")
            return result
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def generate_evaluation_response(self, prompt):
        """
        Generates an evaluation response using the ollama.

        This method adds a system prompt indicating that the response is for evaluation purposes,
        and then sends the prompt to the API for generating a response.

        Args:
            prompt (str): The prompt to evaluate.

        Returns:
            str: The generated evaluation response.
        """
        system_prompt = "You are an assistant for LLM evaluation."
        messages = system_prompt + prompt
        try:
            response = self._generate_response(messages).lower()
            if response.startswith("```json") and response.endswith("```"):
                response = response[7:-3].strip()
            return response
        except Exception as e:
            logger.error(f"Error Generating Evaluation: {e}")
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
        messages = system_prompt + "\n" +prompt
        response = self._generate_response(messages=messages)

        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        return response
