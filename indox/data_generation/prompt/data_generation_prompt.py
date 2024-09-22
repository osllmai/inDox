class DataGenerationPrompt:
    """A class to provide a comprehensive instruction for data generation."""

    @staticmethod
    def get_instruction(user_prompt: str = "") -> str:
        """Return a comprehensive instruction for generating synthetic data,
        focusing on extracting keywords and creating unique data."""
        base_instruction = (
            "You are an advanced AI designed to generate unique and non-repetitive synthetic data."
            " Your task is to first carefully extract relevant keywords from the provided input."
            " Then, based on those keywords, generate a highly unique dataset."
            " Ensure the dataset is creative, never repeated, and entirely original."
            " Output only a dictionary format with no additional text, explanations, or phrases."
            " Avoid using common phrases like 'Here is', 'Generated data', or any similar expressions."
            " Only return the dictionary format without any extra information."
        )

        return f"{base_instruction} {user_prompt}".strip()
