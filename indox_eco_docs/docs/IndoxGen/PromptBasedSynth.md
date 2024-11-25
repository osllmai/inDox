# PromptBasedSynth

## Overview
`PromptBasedSynth` is a Python class designed to generate synthetic data using a Language Learning Model (LLM) based on a prompt and predefined data structure. It allows users to generate data from scratch or augment existing data by feeding a DataFrame to the model. The class can handle both text generation and JSON responses, ensuring that the generated data fits the specified prompt and format.

## Table of Contents
1. [Installation](#installation)
2. [LLM Setup](#llm-setup)
3. [Usage](#usage)
4. [API Reference](#api-reference)
5. [Examples](#examples)
6. [Contributing](#contributing)

## Installation
To use `PromptBasedSynth`, install the required libraries using pip:
```bash
pip install pandas loguru json
```

Additionally, you need the `indoxGen` library to connect to various language models for data generation.

```bash
pip install indoxGen
```

## LLM Setup
`PromptBasedSynth` uses language models (LLMs) to generate synthetic data. The library supports various models via the `indoxGen` library.

For example, you can initialize an `IndoxApi` model like this:
```python
from indoxGen.llms import IndoxApi
import os
from dotenv import load_dotenv

# Load API key from environment variables
load_dotenv()
INDOX_API_KEY = os.getenv("INDOX_API_KEY")

LLM = IndoxApi(api_key=INDOX_API_KEY)
```

The `indoxGen` library supports various models including:
- OpenAI
- Mistral
- Ollama
- Google AI
- Hugging Face models

Additionally, `indoxGen` provides a router for OpenAI, allowing for easy switching between different models.


## Usage
The `PromptBasedSynth` class can be used to either generate new data from a user-provided prompt or augment existing datasets by generating additional rows based on the provided data.

### Example 1: Generate data from scratch
```python
from indoxGen.synthCore import PromptBasedSynth

user_prompt = "Generate a dataset with 3 column and 3 row about soccer."

LLM = IndoxApi(api_key=INDOX_API_KEY)
# instruction = DataGenerationPrompt.get_instruction(user_prompt)

data_generator = PromptBasedSynth(llm=LLM,user_instruction=user_prompt,verbose=1)

generated_df = data_generator.generate_data()

# print(generated_df)
data_generator.save_to_excel("output_dataFromPrompt.xlsx")
```

### Example 2: Generate data using an existing dataset
```python
from indoxGen.synthCore import PromptBasedSynth
from indoxGen.utils import Excel

dataset_file_path = "output_dataFromPrompt.xlsx"

excel_loader = Excel(dataset_file_path) 
df = excel_loader.load()  
user_prompt = " based on given dataset generate one unique row about soccer"
LLM = IndoxApi(api_key=INDOX_API_KEY)

added_row = PromptBasedSynth(llm=LLM, user_instruction=user_prompt, example_data=df, verbose=1).generate_data()
print(added_row)
```

## API Reference

### `PromptBasedSynth`
#### `__init__(self, prompt_name: str, args: dict, outputs: dict, dataframe: pd.DataFrame = None)`
Initializes the `PromptBasedSynth` class.

**Arguments**:

- `prompt_name` (str): The name of the prompt used for data generation.

- `args` (dict): Arguments containing the LLM instance and user instructions.

- `outputs` (dict): Expected output format.

- `dataframe` (pd.DataFrame, optional): Existing DataFrame to augment data from.

#### `run(self) -> pd.DataFrame`
Generates the data and returns a DataFrame.

**Returns**:

- `pd.DataFrame`: A DataFrame containing generated or augmented data.

#### `save_to_excel(self, file_path: str) -> None`
Saves the generated DataFrame to an Excel file.

**Arguments**:

- `file_path` (str): Path to save the Excel file.

## Examples

### Generate New Data from LLM
```python
user_prompt = "Generate a list of 3 planets and their distances from Earth."
LLM = IndoxApi(api_key=INDOX_API_KEY)
instruction = DataGenerationPrompt.get_instruction(user_prompt)

data_generator = PromptBasedSynth(
    prompt_name="Generate Planet Data",
    args={
        "llm": LLM,
        "n": 1,
        "instruction": instruction,
    },
    outputs={"generations": "generate"},
)

generated_df = data_generator.run()
print(generated_df)
data_generator.save_to_excel("planet_data.xlsx")
```

### Augment Existing Data
```python
dataset_file_path = "planet_data.xlsx"

excel_loader = Excel(dataset_file_path)
df = excel_loader.load()

user_prompt = "Add a new planet to the dataset."
instruction = DataGenerationPrompt.get_instruction(user_prompt)

dataset = PromptBasedSynth(
    prompt_name="Augment Planet Data",
    args={
        "llm": LLM,
        "n": 1,
        "instruction": instruction,
    },
    outputs={"generations": "generate"},
    dataframe=df
)

updated_df = dataset.run()
print(updated_df)
dataset.save_to_excel("updated_planet_data.xlsx")
```

## Contributing
Contributions to the `PromptBasedSynth` class are welcome. To contribute:

1. Fork the repository.

2. Create a new branch for your feature.

3. Add your changes and write tests if applicable.

4. Submit a pull request with a clear description of your changes.