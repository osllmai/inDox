# Attribute Prompt

## Overview

`AttributePromptSynth` is a Python class designed to generate synthetic data based on a set of attributes and user instructions. It utilizes language models (LLMs) to generate prompts and retrieve responses that can be saved as a DataFrame or exported to an Excel file.

## Table of Contents

- [Installation](#installation)
- [Language Model Setup](#language-model-setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Examples](#examples)

## Installation

To use the `AttributePromptSynth` class, you need to have Python 3.9+ installed. You can install the `indoxGen` package using pip:

```bash
pip install indoxGen
```

## Language Model Setup

`AttributePromptSynth` requires an LLM (Language Model) for generating responses from provided prompts. The `indoxGen` library provides a unified interface for various language models. Here's how to set up the language model for this class:

```python
from indoxGen.llms import IndoxApi
import os
from dotenv import load_dotenv

load_dotenv()
INDOX_API_KEY = os.getenv("INDOX_API_KEY")

LLM = IndoxApi(api_key=INDOX_API_KEY)
```

The `indoxGen` library supports various models, including:

- OpenAI
- Mistral
- Ollama
- Google AI
- Hugging Face models

Additionally, `indoxGen` provides routing for OpenAI, enabling easy switching between different models.

## Usage

Here's a basic example of how to use the `AttributePromptSynth` class:

```python
from indoxGen.synthCore import AttributePromptSynth

# Define the arguments for generating prompts
args = {
    "instruction": "Generate a {adjective} sentence that is {length}.",
    "attributes": {
        "adjective": ["serious", "funny"],
        "length": ["short", "long"]
    },
    "llm": LLM
}

# Create an instance of AttributePromptSynth
dataset = AttributePromptSynth(prompt_name="ExamplePrompt",
                                   args=args,
                                   outputs={})

# Run the prompt generation
df = dataset.run()

# Display the generated DataFrame
print(df)
```

## API Reference

### `AttributePromptSynth`

```python
def __init__(self, prompt_name: str, args: dict, outputs: dict):
```

Initializes the `AttributePromptSynth` class.

Generates synthetic data based on the attribute setup and returns it as a pandas DataFrame.

Returns:

- A `pandas.DataFrame` containing the generated data.

```python
def save_to_excel(self, file_path: str, df: pd.DataFrame) -> None:
```

Saves the generated DataFrame to an Excel file.

- `file_path` (str): The path where the Excel file will be saved.
- `df` (pd.DataFrame): The DataFrame to be saved.
- Raises: `ValueError` if the DataFrame is empty or cannot be saved.

## Examples

### Generating Data Based on Attributes

```python
from indoxGen.synthCore import DataFromPrompt
from indoxGen.utils import Excel

dataset_file_path = "output_dataFromPrompt.xlsx"

excel_loader = Excel(dataset_file_path)
df = excel_loader.load()
user_prompt = " based on given dataset generate one unique row about soccer"
LLM = IndoxApi(api_key=INDOX_API_KEY)

added_row = DataFromPrompt(llm=LLM, user_instruction=user_prompt, example_data=df, verbose=1).generate_data()
print(added_row)

```

