# Hybrid LLM-GAN: Synthetic Text and Numerical Data Generation Framework

## Overview

**Hybrid LLM-GAN** is a powerful framework that combines **Generative Adversarial Networks (GANs)** for generating synthetic numerical data and **Language Models (LLMs)** for generating synthetic text data. This hybrid approach allows for generating diverse datasets that contain both structured numerical columns (e.g., age, income) and unstructured text columns (e.g., job title, remarks).

The framework leverages **GANs** for structured data generation and **LLMs** for unstructured data generation, ensuring that the two are coherent and contextually aligned.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Example: GAN and LLM Hybrid Pipeline](#example-gan-and-llm-hybrid-pipeline)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

To install the hybrid framework, ensure you have **Python 3.9+**. Install both **IndoxGen-Torch** (for GAN) and the necessary LLM libraries (for text generation):

```bash
pip install indoxgen-torch
pip install openai  # Or any other LLM provider library
pip install python-dotenv  # For managing API keys securely
```

Additionally, make sure you have PyTorch or TensorFlow (depending on your system configuration) installed for the GAN-based part.

---

## Usage

### Example: GAN and LLM Hybrid Pipeline

```python
import pandas as pd
from indoxGen.hybrid_synth import TextTabularSynth, initialize_gan_synth, initialize_llm_synth
from dotenv import load_dotenv
import os

# Load environment variables (API keys for LLM)
load_dotenv('api.env')

INDOX_API_KEY = os.environ['INDOX_API_KEY']
NVIDIA_API_KEY = os.environ['NVIDIA_API_KEY']

from indoxGen.llms import OpenAi, IndoxApi

# Initialize LLMs for text generation
indox = IndoxApi(api_key=INDOX_API_KEY)
nemotron = OpenAi(api_key=NVIDIA_API_KEY, model="nvidia/nemotron-4-340b-instruct", base_url="https://integrate.api.nvidia.com/v1")

# Sample dataset containing both numerical and text data
sample_data = [
    {'age': 25, 'income': 45.5, 'years_of_experience': 3, 'job_title': 'Junior Developer', 'remarks': 'Looking to grow my career.'},
    {'age': 32, 'income': 60.0, 'years_of_experience': 7, 'job_title': 'Developer', 'remarks': 'Experienced professional.'},
    # ... more data entries
]

data = pd.DataFrame(sample_data)

# Define numerical and text columns
numerical_columns = ['age', 'income', 'years_of_experience']
text_columns = ['job_title', 'remarks']

# Initialize LLM setup for text generation
llm_setup = initialize_llm_synth(
    generator_llm=nemotron,
    judge_llm=indox,
    columns=text_columns,
    example_data=sample_data,
    user_instruction="Generate realistic and diverse text data based on the provided numerical context.",
    diversity_threshold=0.4,
    max_diversity_failures=30,
    verbose=1
)

# Initialize GAN setup for numerical data generation
numerical_data = data[numerical_columns]
gan_setup = initialize_gan_synth(
    input_dim=200,
    generator_layers=[128, 256, 512],
    discriminator_layers=[512, 256, 128],
    learning_rate=2e-4,
    batch_size=64,
    epochs=50,
    n_critic=5,
    categorical_columns=[],
    mixed_columns={},
    integer_columns=['age', 'years_of_experience'],
    data=numerical_data
)

# Create the hybrid pipeline for both text and numerical generation
synth_pipeline = TextTabularSynth(tabular=gan_setup, text=llm_setup)

# Generate synthetic data
num_samples = 10
synthetic_data = synth_pipeline.generate(num_samples)

# Preview the generated synthetic data
print(synthetic_data.head())
```

---

## Configuration

### GAN Configuration:
The **TabularGANConfig** class allows customization of the GAN model for numerical data generation. You can modify parameters like the number of layers, learning rates, and batch sizes for fine-tuning.

### LLM Configuration:
In addition to configuring the GAN, you also set up the LLM with the following parameters:
- **generator_llm**: The language model used to generate text (e.g., OpenAI, Nemotron).
- **judge_llm**: A model for judging the quality of generated text.
- **columns**: The columns in the dataset that represent text data.
- **user_instruction**: Custom instruction guiding the LLM to generate relevant and diverse text.

---

## API Reference

### `initialize_gan_synth`
Used to initialize and configure the GAN for generating synthetic numerical data.

### `initialize_llm_synth`
Initializes and configures the LLM for generating synthetic text data.

### `TextTabularSynth`
Combines the GAN and LLM pipelines into one hybrid pipeline for generating synthetic data with both numerical and text fields.

---

## Contributing

Contributions to the **Hybrid LLM-GAN** project are welcome. Follow these steps to contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Add your changes and write tests if applicable.
4. Submit a pull request with a clear description of your changes.

---
