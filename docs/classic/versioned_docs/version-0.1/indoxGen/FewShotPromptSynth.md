# Few Shot Prompt

## Overview
**FewShotPromptSynth** is a Python class designed for generating synthetic data based on few-shot learning examples and user-provided instructions. It utilizes language models to generate diverse datasets, leveraging pre-existing examples for guidance. The class supports outputting the generated data as a pandas DataFrame and allows saving the results to an Excel file.

## Table of Contents
- [Installation](#installation)
- [Language Model Setup](#language-model-setup)
- [Usage](#usage)
- [API Reference](#api-reference)

## Installation
To use **FewShotPromptSynth**, you need to have Python 3.9+ installed. You can install the required package using pip:

```bash
pip install indoxGen
```

## Language Model Setup
**FewShotPromptSynth** requires a language model for generating synthetic data. The `indoxGen` library provides a unified interface for various language models. Here's how to set up a language model for use in the class:

```python
from indoxGen.llms import IndoxApi

# Setup for IndoxApi model
llm = IndoxApi(api_key=INDOX_API_KEY)
```

The **indoxGen** library supports various models, including:
- OpenAI
- Mistral
- Ollama
- Google AI
- Hugging Face models

Additionally, **indoxGen** provides a router for OpenAI, allowing for easy switching between different models.

## Usage
Here's a basic example of how to use **FewShotPromptSynth**:

```python
from indoxGen.synthCore import FewShotPromptSynth
# Define your Language Model (LLM) instance (replace with the actual LLM you're using)
LLM = IndoxApi(api_key=INDOX_API_KEY)

# Define a user prompt for the generation task
user_prompt = "Describe the formation of stars in simple terms. Return the result in JSON format, with the key 'description'."

# Define few-shot examples (input-output pairs) to help guide the LLM
examples = [
    {
        "input": "Describe the process of photosynthesis.",
        "output": "Photosynthesis is the process by which green plants use sunlight to synthesize food from carbon dioxide and water."
    },
    {
        "input": "Explain the water cycle.",
        "output": "The water cycle is the process by which water circulates between the earth's oceans, atmosphere, and land, involving precipitation, evaporation, and condensation."
    }
]

# Create an instance of FewShotPromptSynth using the defined LLM, user prompt, and few-shot examples
data_generator = FewShotPromptSynth(
    llm=LLM,                            # Language model instance (LLM)
    user_instruction=user_prompt,        # Main user instruction or query
    examples=examples,                   # Few-shot input-output examples
    verbose=1,                           # Verbosity level (optional)
    max_tokens=8000                      # Max tokens for generation (optional)
)

# Generate the data based on the few-shot setup
df = data_generator.generate_data()
```

## API Reference

### FewShotPromptSynth

```python
def __init__(self, prompt_name: str, args: dict, outputs: dict, examples: List[Dict[str, str]]):
```
Initializes the **FewShotPromptSynth** class.


```python
def save_to_excel(self, file_path: str, df: pd.DataFrame) -> None:
```
Saves the generated DataFrame to an Excel file.

- `file_path` (str): The path where the Excel file will be saved.
- `df` (pd.DataFrame): The DataFrame to be saved.
- Raises: `ValueError` if the DataFrame is empty or cannot be saved.

