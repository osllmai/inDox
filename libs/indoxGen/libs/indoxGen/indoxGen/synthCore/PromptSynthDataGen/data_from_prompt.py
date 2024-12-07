import pandas as pd
import json
import warnings
from loguru import logger
import sys
from typing import Dict, Any, Optional

warnings.filterwarnings("ignore")

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO")
logger.add(sys.stderr, format="<red>{level}</red>: <level>{message}</level>", level="ERROR")


class PromptBasedSynth:
    """
    Generates output from a given LLM based on user instructions.

    This class uses a language model to generate data based on user instructions,
    focusing on producing a single, comprehensive response.
    """

    def __init__(
            self,
            llm,
            user_instruction: str,
            example_data: Optional[pd.DataFrame] = None,
            verbose: int = 0
    ):
        """
        Initialize the DataFromPrompt class.

        Args:
            llm: Language model for generating data.
            user_instruction: Instruction for data generation.
            example_data: Optional pre-existing DataFrame for context.
            verbose: Verbosity level (0 for minimal output, 1 for detailed feedback).
        """
        self.llm = llm
        self.user_instruction = user_instruction
        self.dataframe = example_data
        self.verbose = verbose
        self.generated_data = None

        logger.info(f"DataFromPrompt initialized with verbose level {verbose}")
        if example_data is not None:
            logger.info(f"Example data provided with {len(example_data)} rows and {len(example_data.columns)} columns")
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
        Generate data based on the user instruction.

        Returns:
            pd.DataFrame: Generated data as a DataFrame.
        """
        logger.info("Starting data generation process")
        generated = self._generate_data_point()
        if generated:
            self.generated_data = generated
            logger.info("Data generated successfully")
            logger.debug(f"Generated data: {json.dumps(generated, indent=2)}")
        else:
            logger.warning("Failed to generate valid data")

        return self.to_dataframe()

    def _generate_data_point(self) -> Dict[str, Any]:
        """Generate a single data point."""
        system_prompt = ("You are an advanced data generator. Create a comprehensive and realistic response based on "
                         "the given instruction. Your response must be a valid JSON object with all property names "
                         "enclosed in double quotes.")
        prompt = self._create_generation_prompt()
        logger.debug(f"Generation prompt: {prompt}")

        for attempt in range(3):
            logger.info(f"Attempt {attempt + 1}/3 to generate data")
            try:
                generated = self.llm.chat(prompt=prompt, system_prompt=system_prompt, max_tokens=8000)
                logger.debug(f"Raw generated text: {generated}")

                # Extract the JSON object
                start = generated.find('{')
                end = generated.rfind('}')
                if start != -1 and end != -1 and start < end:
                    json_str = generated[start:end + 1]
                    json_str = json_str.replace("'", '"')  # Ensure valid JSON format
                    data = json.loads(json_str)
                    logger.info("Successfully extracted and parsed JSON data")
                    return data
                else:
                    logger.warning(f"Failed to find valid JSON object in generated text (Attempt {attempt + 1}/3)")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse generated data (Attempt {attempt + 1}/3): {str(e)}")

            if attempt < 2:
                logger.info(f"Retrying generation (Attempt {attempt + 2}/3)...")

        logger.error("Max attempts reached. Failed to generate valid data.")
        return {}

    def _create_generation_prompt(self) -> str:
        """Create a prompt for the generator LLM."""
        prompt = f"User instruction: {self.user_instruction}\n"
        prompt += ("Generate a comprehensive and structured response based on the instruction. "
                   "The response should be detailed, relevant, and in a format that can be parsed as a JSON object.\n")

        if self.dataframe is not None:
            columns = self.dataframe.columns.tolist()
            sample_row = self.dataframe.to_dict(orient='records')
            prompt += f"\nContext (existing data sample): {json.dumps(sample_row)}\n"
            prompt += f"Only use the following columns for your response: {', '.join(columns)}.\n"
            logger.debug(f"Added context to prompt with {len(columns)} columns")

        prompt += (
            "\nGenerate a single, comprehensive response as a JSON object. Make sure the response contains "
            "only the specified columns and do not add any extra fields. The response should be directly relevant to the user instruction."
        )
        logger.debug("Generation prompt created")
        return prompt

    def to_dataframe(self) -> pd.DataFrame:
        """Convert generated data to a pandas DataFrame, handling various data structures."""
        if self.generated_data is None:
            logger.error("No data has been generated yet. Call generate_data() first.")
            return pd.DataFrame()

        logger.info("Converting generated data to DataFrame")

        if isinstance(self.generated_data, list):
            if all(isinstance(item, dict) for item in self.generated_data):
                df = pd.DataFrame(self.generated_data)
                logger.info(f"Created DataFrame from list of dictionaries: {df.shape}")
                return df

        elif isinstance(self.generated_data, dict):
            for key, value in self.generated_data.items():
                if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                    df = pd.DataFrame(value)
                    logger.info(f"Created DataFrame from nested list of dictionaries: {df.shape}")
                    return df
                elif isinstance(value, dict):
                    df = pd.DataFrame([value])
                    logger.info(f"Created DataFrame from nested dictionary: {df.shape}")
                    return df

            df = pd.DataFrame([self.generated_data])
            logger.info(f"Created DataFrame from single dictionary: {df.shape}")
            return df

        logger.error(f"Unexpected data type: {type(self.generated_data)}. Cannot convert to DataFrame.")
        return pd.DataFrame()

    def save_to_excel(self, file_path: str) -> None:
        """
        Saves the generated data to an Excel file.

        Args:
            file_path (str): The path where the Excel file will be saved.

        Raises:
            ValueError: If no data has been generated or it cannot be saved.
        """
        df = self.to_dataframe()

        if df.empty:
            logger.error("No data to save. Generate data first.")
            raise ValueError("No data to save. Generate data first.")

        try:
            df.to_excel(file_path, index=False)
            logger.info(f"Data saved to Excel file at: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save data to Excel: {e}")
            raise ValueError(f"Failed to save data to Excel: {e}")
