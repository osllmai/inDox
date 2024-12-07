import re
import pandas as pd
import json
import warnings
from loguru import logger
import sys
from typing import List, Dict, Any

warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO")
logger.add(sys.stderr, format="<red>{level}</red>: <level>{message}</level>", level="ERROR")


class FewShotPromptSynth:
    """
    Generates outputs from a given LLM based on few-shot examples and a user setup.
    """

    def __init__(
            self,
            llm: Any,
            user_instruction: str,
            examples: List[Dict[str, str]],
            verbose: int = 0,
            max_tokens: int = 8000
    ):
        """
        Initializes the FewShotPrompt class with the prompt configuration.

        Args:
            llm (Any): The language model instance.
            user_instruction (str): The main user instruction or query.
            examples (List[Dict[str, str]]): List of few-shot input-output examples.
            verbose (int, optional): Verbosity level for logging. Defaults to 0.
            max_tokens (int, optional): Maximum tokens for LLM output. Defaults to 8000.
        """
        self.llm = llm
        self.user_instruction = user_instruction
        self.examples = examples
        self.verbose = verbose
        self.max_tokens = max_tokens
        logger.info(f"FewShotPrompt initialized with {len(examples)} examples and max_tokens={max_tokens}")

    def prepare_prompt(self) -> str:
        """
        Prepares the full prompt including few-shot examples and the current input.

        Returns:
            str: The formatted prompt to send to the LLM.
        """
        few_shot_examples = ""

        # Prepare few-shot examples
        for example in self.examples:
            input_example = example.get("input", "")
            output_example = example.get("output", "")
            few_shot_examples += f"Input: {input_example}\nOutput: {output_example}\n\n"

        # Include the user instruction as the final input in the prompt
        full_prompt = f"{few_shot_examples}Input: {self.user_instruction}\nOutput:"

        logger.debug(f"Prepared prompt with {len(self.examples)} examples")
        return full_prompt

    def generate_data(self) -> pd.DataFrame:
        """
        Generates data based on the few-shot setup and returns a DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the generated results.
        """
        full_prompt = self.prepare_prompt()
        logger.info("Generating data from LLM")

        try:
            # Send the prompt to the LLM and receive the response
            generations = self.llm.chat(prompt=full_prompt, max_tokens=self.max_tokens)
            logger.debug("Successfully received response from LLM")
        except Exception as e:
            logger.error(f"Failed to generate text from LLM: {e}")
            raise ValueError(f"Failed to generate text from LLM: {e}")

        # Handle if the response is a list or a single string
        full_response = ''.join(generations) if isinstance(generations, list) else generations

        # Check if the response is empty
        if not full_response.strip():
            logger.error("Received an empty response from the LLM.")
            raise ValueError("Received an empty response from the LLM.")

        # Clean the response
        cleaned_response = self.clean_llm_response(full_response)

        try:
            # Attempt to parse the cleaned LLM response as JSON
            results = json.loads(cleaned_response)
            logger.info("Successfully parsed JSON response")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}. Attempting to extract JSON from markdown.")
            try:
                # Try to extract JSON from markdown code block
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', cleaned_response)
                if json_match:
                    results = json.loads(json_match.group(1))
                    logger.info("Successfully extracted JSON from markdown")
                else:
                    logger.error("No JSON found in markdown. Treating as plain text.")
                    results = [cleaned_response]
            except json.JSONDecodeError:
                logger.error("Failed to extract valid JSON from markdown. Treating as plain text.")
                results = [cleaned_response]

        # Create a DataFrame from the results (handle both dict and list responses)
        if isinstance(results, dict):
            df = pd.DataFrame([results])
        else:
            df = pd.DataFrame(results, columns=["output"])

        logger.info(f"Created DataFrame with {len(df)} rows")
        return df

    def clean_llm_response(self, response: str) -> str:
        """
        Cleans the LLM response by removing unnecessary characters and markdown syntax.

        Args:
            response (str): The raw response from the LLM.

        Returns:
            str: The cleaned response.
        """
        logger.debug("Cleaning LLM response")
        # Remove markdown code block syntax
        cleaned = re.sub(r'```json\s*|\s*```', '', response)

        # Remove escaped newlines and quotes
        cleaned = re.sub(r'\\n|\\"|\\\'', '', cleaned)

        # Remove leading/trailing whitespace
        cleaned = cleaned.strip()

        logger.debug("LLM response cleaned")
        return cleaned

    def save_to_excel(self, file_path: str, df: pd.DataFrame) -> None:
        """
        Saves the generated DataFrame to an Excel file.

        Args:
            file_path (str): The path where the Excel file will be saved.
            df (pd.DataFrame): The DataFrame to be saved.

        Raises:
            ValueError: If the DataFrame is empty or cannot be saved.
        """
        if df.empty:
            logger.error("DataFrame is empty. Cannot save to Excel.")
            raise ValueError("DataFrame is empty. Cannot save to Excel.")

        try:
            df.to_excel(file_path, index=False)
            logger.info(f"DataFrame saved to Excel file at: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save DataFrame to Excel: {e}")
            raise ValueError(f"Failed to save DataFrame to Excel: {e}")
