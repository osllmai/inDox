# Interactive Feedback

## Overview

`InteractiveFeedbackSynth` is a Python class designed for generating synthetic data based on example data and user instructions. It utilizes language models to generate and judge synthetic data points, ensuring diversity and adherence to specified criteria. This class is particularly useful for creating realistic, diverse datasets for testing, development, or machine learning purposes. It also includes a human feedback mechanism for reviewing and regenerating data points.

## Table of Contents

- [Installation](#installation)
- [Language Model Setup](#language-model-setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Examples](#examples)

## Installation

To use the `InteractiveFeedbackSynth`, you need to have Python 3.9+ installed. You can install the required dependencies using pip:

```bash
pip install indoxGen
```

## Language Model Setup

The `InteractiveFeedbackSynth` requires two language models: one for generating data and another for judging data quality. The specific setup will depend on your chosen language model library. Here's an example using a hypothetical library:

```python
from indoxGen.llms import IndoxApi, OpenAi

# generator llm
indox = IndoxApi(api_key=INDOX_API_KEY)

#judge llm
nemotron = OpenAi(api_key=NVIDIA_API_KEY, model="nvidia/nemotron-4-340b-instruct",
                  base_url="https://integrate.api.nvidia.com/v1")
```

When initializing the `InteractiveFeedbackSynth`, you'll pass these language model instances as the `generator_llm` and `judge_llm` parameters.

## Usage

Here's a basic example of how to use the `InteractiveFeedbackSynth`:

```python
from indoxGen.synthCore import InteractiveFeedbackSynth
from indoxGen.llms import OpenAi, IndoxApi

# Setup language models
generator_llm = IndoxApi(api_key=INDOX_API_KEY)
judge_llm = OpenAi(api_key=NVIDIA_API_KEY, model="nvidia/nemotron-4-340b-instruct",
                   base_url="https://integrate.api.nvidia.com/v1")

columns = ["name", "age", "occupation"]
example_data = [
    {"name": "John Doe", "age": 30, "occupation": "Engineer"},
    {"name": "Jane Smith", "age": 28, "occupation": "Teacher"}
]
user_instruction = "Generate diverse fictional employee data"

generator = InteractiveFeedbackSynth(
    generator_llm=generator_llm,
    judge_llm=judge_llm,
    columns=columns,
    example_data=example_data,
    user_instruction=user_instruction,
    feedback_min_score= 0.8
)

synthetic_data = generator.generate_data(num_samples=10)
print(synthetic_data)

# User review and regeneration
accepted_rows = [0, 1, 2]  # Indices of rows to accept
regenerate_rows = [3, 4]  # Indices of rows to regenerate
regeneration_feedback = "Ensure more diversity in occupations"
updated_data = generator.user_review_and_regenerate(accepted_rows, regenerate_rows, regeneration_feedback, min_score=0.7)
print(updated_data)
```

## API Reference

### InteractiveFeedbackSynth

#### `__init__(self, generator_llm, judge_llm, columns, example_data, user_instruction, real_data=None, diversity_threshold=0.7, max_diversity_failures=20, verbose=0, feedback_min_score=0.8)`

Initialize the InteractiveFeedbackSynth.

- `generator_llm`: Language model for generating data.
- `judge_llm`: Language model for judging data quality.
- `columns`: List of column names for the synthetic data.
- `example_data`: List of example data points.
- `user_instruction`: Instruction for data generation.
- `real_data`: Optional list of real data points.
- `diversity_threshold`: Threshold for determining data diversity (default: 0.7).
- `max_diversity_failures`: Maximum number of diversity failures before forcing acceptance (default: 20).
- `verbose`: Verbosity level (0 for minimal output, 1 for detailed feedback) (default: 0).
- `feedback_min_score`: Minimum score for accepting generated data (default: 0.8).

#### `generate_data(self, num_samples: int) -> pd.DataFrame`

Generate synthetic data points.

- `num_samples`: Number of data points to generate.
- Returns: DataFrame containing the generated data.

#### `user_review_and_regenerate(self, accepted_rows: Union[List[int], List[str]], regenerate_rows: Union[List[int], List[str]], regeneration_feedback: str, min_score: float) -> pd.DataFrame`

Review and regenerate synthetic data based on feedback.

- `accepted_rows`: Indices of rows to accept or ['all'].
- `regenerate_rows`: Indices of rows to regenerate or ['all'].
- `regeneration_feedback`: Feedback for regeneration.
- `min_score`: Minimum score for accepting regenerated data.
- Returns: Generated dataframe containing accepted and regenerated data.

## Examples

### Generating Employee Data with Human Feedback

```python
columns = ["name", "age", "department", "salary"]
example_data = [
    {"name": "Alice Johnson", "age": 35, "department": "Marketing", "salary": 75000},
    {"name": "Bob Williams", "age": 42, "department": "Engineering", "salary": 90000}
]
user_instruction = "Generate diverse employee data for a tech company"

generator = InteractiveFeedbackSynth(
    generator_llm=your_generator_llm,
    judge_llm=your_judge_llm,
    columns=columns,
    example_data=example_data,
    user_instruction=user_instruction,
    verbose=1,
    feedback_min_score= 0.8
)

employee_data = generator.generate_data(num_samples=50)
print(employee_data.head())
# User review and regeneration
accepted_rows = [0, 1, 2]  # Indices of rows to accept
regenerate_rows = [3, 4]  # Indices of rows to regenerate
regeneration_feedback = "Ensure more diversity in department assignments and a wider range of salaries"
updated_data = generator.user_review_and_regenerate(accepted_rows, regenerate_rows, regeneration_feedback, min_score=0.7)
print(updated_data.head())
```
