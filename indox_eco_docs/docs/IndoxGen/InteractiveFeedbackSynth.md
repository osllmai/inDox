# InteractiveFeedbackSynth

## Overview

`InteractiveFeedbackSynth` is a Python class designed to generate synthetic data based on example data and user instructions. It uses language models for data generation and quality assessment, ensuring the output is diverse, realistic, and adheres to specified criteria. The class also provides a human feedback mechanism for reviewing and regenerating data points.

---

## Table of Contents
1. [Installation](#installation)
2. [Language Model Setup](#language-model-setup)
3. [Usage](#usage)
4. [API Reference](#api-reference)
5. [Examples](#examples)
6. [Contributing](#contributing)

---

## Installation

To use `InteractiveFeedbackSynth`, you need Python 3.9+ installed. Install the required dependencies using pip:

```bash
pip install indoxGen
```

---

## Language Model Setup

`InteractiveFeedbackSynth` requires two language models:
- **Generator LLM**: For generating synthetic data.
- **Judge LLM**: For assessing the quality of generated data.

Here’s an example setup:

```python
from indoxGen.llms import IndoxApi, OpenAi

# Initialize the generator LLM
generator_llm = IndoxApi(api_key=INDOX_API_KEY)

# Initialize the judge LLM
judge_llm = OpenAi(
    api_key=NVIDIA_API_KEY,
    model="nvidia/nemotron-4-340b-instruct",
    base_url="https://integrate.api.nvidia.com/v1"
)
```

These models are passed as `generator_llm` and `judge_llm` to `InteractiveFeedbackSynth`.

---

## Usage

### Example: Basic Workflow

Below is an example workflow for generating synthetic data with `InteractiveFeedbackSynth`:

```python
from indoxGen.synthCore import InteractiveFeedbackSynth
from indoxGen.llms import IndoxApi, OpenAi

# Setup language models
generator_llm = IndoxApi(api_key=INDOX_API_KEY)
judge_llm = OpenAi(
    api_key=NVIDIA_API_KEY,
    model="nvidia/nemotron-4-340b-instruct",
    base_url="https://integrate.api.nvidia.com/v1"
)

# Define data structure
columns = ["name", "age", "occupation"]
example_data = [
    {"name": "John Doe", "age": 30, "occupation": "Engineer"},
    {"name": "Jane Smith", "age": 28, "occupation": "Teacher"}
]
user_instruction = "Generate diverse fictional employee data"

# Initialize InteractiveFeedbackSynth
synth = InteractiveFeedbackSynth(
    generator_llm=generator_llm,
    judge_llm=judge_llm,
    columns=columns,
    example_data=example_data,
    user_instruction=user_instruction,
    feedback_min_score=0.8
)

# Generate synthetic data
synthetic_data = synth.generate_data(num_samples=10)
print(synthetic_data)

# Review and regenerate data
accepted_rows = [0, 1, 2]
regenerate_rows = [3, 4]
feedback = "Ensure more diversity in occupations"
updated_data = synth.user_review_and_regenerate(
    accepted_rows,
    regenerate_rows,
    feedback,
    min_score=0.7
)
print(updated_data)
```

---

## API Reference

### `InteractiveFeedbackSynth`

#### `__init__`
Initialize the `InteractiveFeedbackSynth` class.

**Parameters:**

- `generator_llm`: Language model for data generation.

- `judge_llm`: Language model for assessing data quality.

- `columns`: List of column names for the synthetic data.

- `example_data`: Example dataset as a list of dictionaries.

- `user_instruction`: Instruction for generating data.

- `real_data` *(optional)*: List of real data points.

- `diversity_threshold` *(default: 0.7)*: Threshold for determining data diversity.

- `max_diversity_failures` *(default: 20)*: Maximum allowed diversity failures.

- `verbose` *(default: 0)*: Verbosity level (0 for minimal output).

- `feedback_min_score` *(default: 0.8)*: Minimum score for accepting generated data.

---

#### `generate_data`
Generates synthetic data points.

**Parameters:**

- `num_samples`: Number of synthetic data points to generate.

**Returns:**

- A Pandas DataFrame containing the generated data.

---

#### `user_review_and_regenerate`

Allows users to review and regenerate synthetic data based on feedback.

**Parameters:**

- `accepted_rows`: List of indices for rows to accept (or `['all']`).

- `regenerate_rows`: List of indices for rows to regenerate (or `['all']`).

- `regeneration_feedback`: Feedback for the regeneration process.

- `min_score`: Minimum score for accepting regenerated data.

**Returns:**

- A Pandas DataFrame containing the updated data.

---

## Examples

### Generating Employee Data with Feedback

```python
columns = ["name", "age", "department", "salary"]
example_data = [
    {"name": "Alice Johnson", "age": 35, "department": "Marketing", "salary": 75000},
    {"name": "Bob Williams", "age": 42, "department": "Engineering", "salary": 90000}
]
user_instruction = "Generate diverse employee data for a tech company"

# Initialize InteractiveFeedbackSynth
generator = InteractiveFeedbackSynth(
    generator_llm=your_generator_llm,
    judge_llm=your_judge_llm,
    columns=columns,
    example_data=example_data,
    user_instruction=user_instruction,
    verbose=1,
    feedback_min_score=0.8
)

# Generate synthetic data
employee_data = generator.generate_data(num_samples=50)
print(employee_data.head())

# Review and regenerate data
accepted_rows = [0, 1, 2]
regenerate_rows = [3, 4]
feedback = "Ensure diversity in departments and wider salary ranges"
updated_data = generator.user_review_and_regenerate(
    accepted_rows,
    regenerate_rows,
    feedback,
    min_score=0.7
)
print(updated_data.head())
```

---

## Contributing

Contributions to improve `InteractiveFeedbackSynth` are welcome. Follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Add your changes and write tests, if applicable.
4. Submit a pull request with a detailed description of your changes.

When contributing, maintain the existing code style and add proper documentation for new features.
