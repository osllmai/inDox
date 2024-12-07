import pandas as pd
import json
from loguru import logger
import sys
from itertools import product
from typing import List, Dict, Any

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO")
logger.add(sys.stderr, format="<red>{level}</red>: <level>{message}</level>", level="ERROR")


class AttributePromptSynth:
    """
    Generates data from a given LLM based on attributes and user instructions,
    focusing on concise sentence output.
    """

    def __init__(
            self,
            llm: Any,
            user_instruction: str,
            attributes: Dict[str, List[str]],
            verbose: int = 0,
            max_tokens: int = 8000
    ):
        """
        Initialize the DataFromAttributedPrompt.

        Args:
            llm: The language model for generating data.
            user_instruction: Instruction template for generating data.
            attributes: A dictionary of attribute options to generate combinations.
            verbose: Verbosity level (0 for minimal output, 1 for detailed feedback).
            max_tokens: Maximum tokens for the LLM response.
        """
        self.llm = llm
        self.user_instruction = user_instruction
        self.attributes = attributes
        self.verbose = verbose
        self.max_tokens = max_tokens
        self.generated_data = []

        logger.info(
            f"DataFromAttributedPrompt initialized with {len(attributes)} attributes and max_tokens={max_tokens}")
        if self.verbose >= 1:
            logger.debug(f"Attributes: {attributes}")
            logger.debug(f"User instruction: {user_instruction}")

    @staticmethod
    def get_instruction(user_prompt: str = "") -> str:
        """
        Return a comprehensive instruction for generating data.

        Args:
            user_prompt: User-provided prompt for data generation.

        Returns:
            str: Comprehensive instruction for data generation.
        """
        base_instruction = (
            "You are an advanced AI designed to generate unique and comprehensive data."
            " Your task is to carefully extract relevant information from the provided input"
            " and generate a highly detailed and structured response."
            " Ensure the output is creative, relevant, and entirely original."
            " The response should be in a format that can be parsed as a JSON object."
            " Avoid using common phrases like 'Here is', 'Generated data', or any similar expressions."
            " Only return the structured data without any additional text or explanations."
        )
        instruction = f"{base_instruction} {user_prompt}".strip()
        logger.debug(f"Generated instruction: {instruction}")
        return instruction

    def generate_data(self) -> pd.DataFrame:
        """
        Generate data points based on attribute combinations and user instructions.

        Returns:
            DataFrame containing the generated data.
        """
        logger.info("Starting data generation process")
        prompts = self._prepare_prompts()
        results = []

        for prompt in prompts:
            logger.debug(f"Generating data for prompt: {prompt}")
            generated_data = self._generate_single_data_point(prompt)
            if generated_data:
                if isinstance(generated_data, dict) and 'sentence' in generated_data:
                    results.append(generated_data['sentence'])
                else:
                    results.append(generated_data)

        if len(results) == 0:
            logger.error("No data generated.")
            return pd.DataFrame()

        df = pd.DataFrame(results, columns=["sentence"])
        logger.info(f"Generated {len(df)} data points")
        return df

    def _prepare_prompts(self) -> List[str]:
        """
        Prepares prompts by combining attributes with the user instruction.

        Returns:
            List of formatted prompts.
        """
        logger.info("Preparing prompts from attributes")
        attribute_combinations = product(*self.attributes.values())
        prompts = []

        for combination in attribute_combinations:
            attribute_dict = dict(zip(self.attributes.keys(), combination))
            instruction = self.get_instruction(self.user_instruction.format(**attribute_dict))
            prompts.append(instruction)

        logger.info(f"Generated {len(prompts)} prompts from attributes")
        return prompts

    def _generate_single_data_point(self, prompt: str) -> Any:
        """
        Generate a single data point from the language model.

        Args:
            prompt: The input prompt for the LLM.

        Returns:
            A string or sentence extracted from the generated response.
        """
        try:
            logger.debug("Sending prompt to LLM")
            response = self.llm.chat(prompt=prompt, max_tokens=self.max_tokens)
            if isinstance(response, list):
                response = ''.join(response)

            try:
                result = json.loads(response)
                logger.debug("Successfully parsed JSON result")
                extracted_data = self._extract_data(result)
                logger.debug(f"Extracted data: {extracted_data}")
                return extracted_data

            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON. Using raw response as fallback: {response}")
                return response

        except Exception as e:
            logger.error(f"Failed to generate data from LLM: {e}")
            return prompt

    def _extract_data(self, data: Any) -> str:
        """
        Recursively extract meaningful text from various types of data structures.

        Args:
            data: The data to extract from, can be a dict, list, or simple string.

        Returns:
            A string with the extracted information.
        """
        logger.debug(f"Extracting data from: {type(data)}")
        if isinstance(data, dict):
            for key, value in data.items():
                extracted_value = self._extract_data(value)
                if extracted_value:
                    return extracted_value

        elif isinstance(data, list):
            for item in data:
                extracted_value = self._extract_data(item)
                if extracted_value:
                    return extracted_value

        elif isinstance(data, str):
            return data

        logger.warning("No meaningful data extracted")
        return ""
