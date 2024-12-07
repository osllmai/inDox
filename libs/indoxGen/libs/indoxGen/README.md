
# IndoxGen: Enterprise-Grade Synthetic Data Generation Framework


[![License](https://img.shields.io/github/license/osllmai/indoxGen)](https://github.com/osllmai/IndoxGen/tree/master/libs/indoxGen/LICENSE)
[![PyPI](https://badge.fury.io/py/indoxGen.svg)](https://pypi.org/project/indoxGen/0.1.0/)
[![Python](https://img.shields.io/pypi/pyversions/indoxGen.svg)](https://pypi.org/project/indoxGen/0.1.0/)
[![Downloads](https://static.pepy.tech/badge/indoxGen)](https://pepy.tech/project/indoxGen)

[![Discord](https://img.shields.io/discord/1223867382460579961?label=Discord&logo=Discord&style=social)](https://discord.com/invite/ossllmai)
[![GitHub stars](https://img.shields.io/github/stars/osllmai/indoxGen?style=social)](https://github.com/osllmai/indoxGen)

<p align="center">
  <a href="https://osllm.ai">Official Website</a> &bull; <a href="https://docs.osllm.ai/index.html">Documentation</a> &bull; <a href="https://discord.gg/qrCc56ZR">Discord</a>
</p>

<p align="center">
  <b>NEW:</b> <a href="https://docs.google.com/forms/d/1CQXJvxLUqLBSXnjqQmRpOyZqD6nrKubLz2WTcIJ37fU/prefill">Subscribe to our mailing list</a> for updates and news!
</p>


## Overview

IndoxGen is a state-of-the-art, enterprise-ready framework designed for generating high-fidelity synthetic data. Leveraging advanced AI technologies, including Large Language Models (LLMs) and incorporating human feedback loops, IndoxGen offers unparalleled flexibility and precision in synthetic data creation across various domains and use cases.

## Key Features

- **Multiple Generation Pipelines**:
  - `SyntheticDataGenerator`: Standard LLM-powered generation pipeline for structured data with embedded quality control mechanisms.
  - `SyntheticDataGeneratorHF`: Advanced pipeline integrating human feedback to improve generation.
  - `DataFromPrompt`: Dynamic data generation based on natural language prompts, useful for rapid prototyping.
  
- **Customization & Control**: Fine-grained control over data attributes, structure, and diversity. Customize every aspect of the synthetic data generation process.
  
- **Human-in-the-Loop**: Seamlessly integrates expert feedback for continuous improvement of generated data, offering the highest quality assurance.
  
- **AI-Driven Diversity**: Algorithms ensure representative and varied datasets, providing data diversity for robust modeling.
  
- **Flexible I/O**: Supports various data sources and export formats (Excel, CSV, etc.) for easy integration into existing workflows.
  
- **Advanced Learning Techniques**: Incorporation of few-shot learning for rapid adaptation to new domains with minimal examples.
  
- **Scalability**: Designed to handle both small-scale experiments and large-scale data generation tasks with multi-LLM support.

## Installation

```bash
pip install indoxgen
```

## Quick Start Guide

### Basic Usage: SyntheticDataGenerator

```python
from indoxGen.synthCore import SyntheticDataGenerator
from indoxGen.llms import OpenAi

columns = ["name", "age", "occupation"]
example_data = [
    {"name": "Alice Johnson", "age": 35, "occupation": "Manager"},
    {"name": "Bob Williams", "age": 42, "occupation": "Accountant"}
]

openai = OpenAi(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
nemotron = OpenAi(api_key=NVIDIA_API_KEY, model="nvidia/nemotron-4-340b-instruct",
                  base_url="https://integrate.api.nvidia.com/v1")

generator = SyntheticDataGenerator(
    generator_llm=nemotron,
    judge_llm=openai,
    columns=columns,
    example_data=example_data,
    user_instruction="Generate diverse, realistic data including name, age, and occupation. Ensure variability in demographics and professions.",
    verbose=1
)

generated_data = generator.generate_data(num_samples=100)
```

### Advanced Usage: SyntheticDataGeneratorHF with Human Feedback

```python
from indoxGen.synthCore import SyntheticDataGeneratorHF
from indoxGen.llms import OpenAi

openai = OpenAi(api_key=OPENAI_API_KEY, model="gpt-4-0613")
nemotron = OpenAi(api_key=NVIDIA_API_KEY, model="nvidia/nemotron-4-340b-instruct",
                  base_url="https://integrate.api.nvidia.com/v1")

generator = SyntheticDataGeneratorHF(
    generator_llm=nemotron,
    judge_llm=openai,
    columns=columns,
    example_data=example_data,
    user_instruction="Generate diverse, realistic professional profiles with name, age, and occupation.",
    verbose=1,
    diversity_threshold=0.4,
    feedback_range=feedback_range
)

# Implement human feedback loop
generator.user_review_and_regenerate(
    regenerate_rows=[0],
    accepted_rows=[],
    regeneration_feedback='Diversify names and occupations further',
    min_score=0.7
)
```

### Prompt-Based Generation: DataFromPrompt

```python
from indoxGen.synthCore import DataFromPrompt, DataGenerationPrompt
from indoxGen.llms import OpenAi

nemotron = OpenAi(api_key=NVIDIA_API_KEY, model="nvidia/nemotron-4-340b-instruct",
                  base_url="https://integrate.api.nvidia.com/v1")


user_prompt = "Generate a comprehensive dataset with 3 columns and 3 rows about exoplanets."
instruction = DataGenerationPrompt.get_instruction(user_prompt)

data_generator = DataFromPrompt(
    prompt_name="Exoplanet Dataset Generation",
    args={
        "llm": nemotron,
        "n": 1,
        "instruction": instruction,
    },
    outputs={"generations": "generate"},
)

generated_df = data_generator.run()
data_generator.save_to_excel("exoplanet_data.xlsx")
```

## Advanced Techniques

### Few-Shot Learning for Specialized Domains

```python
from indoxGen.synthCore import FewShotPrompt
from indoxGen.llms import OpenAi

openai = OpenAi(api_key=OPENAI_API_KEY, model="gpt-4o-mini")

examples = [
    {
        "input": "Generate a dataset with 3 columns and 2 rows about quantum computing.",
        "output": '[{"Qubit Type": "Superconducting", "Coherence Time": "100 μs", "Gate Fidelity": "0.9999"}, {"Qubit Type": "Trapped Ion", "Coherence Time": "10 ms", "Gate Fidelity": "0.99999"}]'
    },
    {
        "input": "Generate a dataset with 3 columns and 2 rows about nanotechnology.",
        "output": '[{"Material": "Graphene", "Thickness": "1 nm", "Conductivity": "1.0e6 S/m"}, {"Material": "Carbon Nanotube", "Thickness": "1-2 nm", "Conductivity": "1.0e7 S/m"}]'
    }
]

user_prompt = "Generate a dataset with 3 columns and 2 rows about advanced AI architectures."

data_generator = FewShotPrompt(
    prompt_name="Generate AI Architecture Dataset",
    args={
        "llm": openai,
        "n": 1,  
        "instruction": user_prompt,  
    },
    outputs={"generations": "generate"},
    examples=examples  
)

generated_df = data_generator.run()
data_generator.save_to_excel("ai_architectures.xlsx", generated_df)
```

### Attributed Prompts for Controlled Variation

```python
from indoxGen.synthCore import DataFromAttributedPrompt
from indoxGen.llms import OpenAi

openai = OpenAi(api_key=OPENAI_API_KEY, model="gpt-4o-mini")

args = {
    "instruction": "Generate a {complexity} machine learning algorithm description that is {application_area} focused.",
    "attributes": {
        "complexity": ["basic", "advanced", "cutting-edge"],
        "application_area": ["computer vision", "natural language processing", "reinforcement learning"]
    },
    "llm": openai
}

dataset = DataFromAttributedPrompt(
    prompt_name="ML Algorithm Generator",
    args=args,
    outputs={}
)

df = dataset.run()
print(df)
```

## Configuration and Customization

Each generator class in IndoxGen is highly configurable to meet specific data generation requirements. Key parameters include:

- `generator_llm` and `judge_llm`: Specify the LLMs used for generation and quality assessment
- `columns` and `example_data`: Define the structure and provide examples for the generated data
- `user_instruction`: Customize the generation process with specific guidelines
- `diversity_threshold`: Control the level of variation in the generated data
- `verbose`: Adjust the level of feedback during the generation process

Refer to the API documentation for a comprehensive list of configuration options for each class.

## Best Practices

1. **Data Quality Assurance**: Regularly validate generated data against predefined quality metrics.
2. **Iterative Refinement**: Utilize the human feedback loop to continuously improve generation quality.
3. **Domain Expertise Integration**: Collaborate with domain experts to fine-tune generation parameters and validate outputs.
4. **Ethical Considerations**: Ensure generated data adheres to privacy standards and ethical guidelines.
5. **Performance Optimization**: Monitor and optimize generation pipeline for large-scale tasks.

## Roadmap
* [x] Implement basic synthetic data generation
* [x] Add LLM-based judge for quality control
* [x] Improve diversity checking mechanism
* [x] Integrate human feedback loop for continuous improvement
* [ ] Develop a web-based UI for easier interaction
* [ ] Support for more data types (images, time series, etc.)
* [ ] Implement differential privacy techniques
* [ ] Create plugin system for custom data generation rules
* [ ] Develop comprehensive documentation and tutorials

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to get started.

## License

IndoxGen is released under the MIT License. See [LICENSE.md](LICENSE.md) for more details.


---

IndoxGen - Empowering Data-Driven Innovation with Advanced Synthetic Data Generation
