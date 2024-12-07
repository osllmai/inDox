# GenerativeDataSynth

## Overview

`GenerativeDataSynth` is a Python class designed for generating synthetic data based on example data and user instructions. It utilizes language models to generate and judge synthetic data points, ensuring diversity and adherence to specified criteria.

## Table of Contents

- [Installation](#installation)
- [Language Model Setup](#language-model-setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Installation

To use the `GenerativeDataSynth`, you need to have Python 3.9+ installed. You can install the package using pip:

```bash
pip install indoxGen
```

## Language Model Setup

The `GenerativeDataSynth` requires two language models: one for generating data and another for judging data quality. The `indoxGen` library provides a unified interface for various language models. Here's how to set up the language models:

```python
from indoxGen.llms import OpenAi

# Setup for OpenAI model
openai = OpenAi(api_key=OPENAI_API_KEY, model="gpt-4-mini")

# Setup for NVIDIA model
nemotron = OpenAi(api_key=NVIDIA_API_KEY, model="nvidia/nemotron-4-340b-instruct",
                  base_url="https://integrate.api.nvidia.com/v1")
```

The `indoxGen` library supports various models including:
- OpenAI
- Mistral
- Ollama
- Google AI
- Hugging Face models

Additionally, `indoxGen` provides a router for OpenAI, allowing for easy switching between different models.

When initializing the `GenerativeDataSynth`, you'll pass these language model instances as the `generator_llm` and `judge_llm` parameters.

## Usage

Here's a basic example of how to use the `GenerativeDataSynth`:

```python
from indoxGen.synthCore import GenerativeDataSynth
from indoxGen.llms import OpenAi

# Setup language models
generator_llm = OpenAi(api_key=OPENAI_API_KEY, model="gpt-4-mini")
judge_llm = OpenAi(api_key=NVIDIA_API_KEY, model="nvidia/nemotron-4-340b-instruct",
                   base_url="https://integrate.api.nvidia.com/v1")

columns = ["name", "age", "occupation"]
example_data = [
    {"name": "John Doe", "age": 30, "occupation": "Engineer"},
    {"name": "Jane Smith", "age": 28, "occupation": "Teacher"}
]
user_instruction = "Generate diverse fictional employee data"

generator = GenerativeDataSynth(
    generator_llm=generator_llm,
    judge_llm=judge_llm,
    columns=columns,
    example_data=example_data,
    user_instruction=user_instruction
)

synthetic_data = generator.generate_data(num_samples=100)
print(synthetic_data)
```

## API Reference

### GenerativeDataSynth

#### `__init__(self, generator_llm, judge_llm, columns, example_data, user_instruction, real_data=None, diversity_threshold=0.7, max_diversity_failures=20, verbose=0)`

Initialize the GenerativeDataSynth.

- `generator_llm`: Language model for generating data.
- `judge_llm`: Language model for judging data quality.
- `columns`: List of column names for the synthetic data.
- `example_data`: List of example data points.
- `user_instruction`: Instruction for data generation.
- `real_data`: Optional list of real data points.
- `diversity_threshold`: Threshold for determining data diversity (default: 0.7).
- `max_diversity_failures`: Maximum number of diversity failures before forcing acceptance (default: 20).
- `verbose`: Verbosity level (0 for minimal output, 1 for detailed feedback) (default: 0).

#### `generate_data(self, num_samples: int) -> pd.DataFrame`

Generate synthetic data points.

- `num_samples`: Number of data points to generate.
- Returns: DataFrame containing the generated data.

## Examples

### Generating Employee Data

```python
columns = ["name", "age", "department", "salary"]
example_data = [
    {"name": "Alice Johnson", "age": 35, "department": "Marketing", "salary": 75000},
    {"name": "Bob Williams", "age": 42, "department": "Engineering", "salary": 90000}
]
user_instruction = "Generate diverse employee data for a tech company"

generator = GenerativeDataSynth(
    generator_llm=your_generator_llm,
    judge_llm=your_judge_llm,
    columns=columns,
    example_data=example_data,
    user_instruction=user_instruction,
    verbose=1
)

employee_data = generator.generate_data(num_samples=50)
print(employee_data.head())
```

## Contributing

Contributions to improve `GenerativeDataSynth` are welcome. Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature.
3. Add your changes and write tests if applicable.
4. Submit a pull request with a clear description of your changes.


