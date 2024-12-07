# IndoxGen-Torch: Advanced GAN-based Synthetic Data Generation Framework

[![License](https://img.shields.io/github/license/osllmai/IndoxGen-Torch)](https://github.com/osllmai/IndoxGen/tree/master/libs/indoxGen_torch/LICENSE)
[![PyPI](https://badge.fury.io/py/IndoxGen-Torch.svg)](https://pypi.org/project/indoxGen-torch/)
[![Python](https://img.shields.io/pypi/pyversions/IndoxGen-Torch.svg)](https://pypi.org/project/indoxGen-torch/)
[![Downloads](https://static.pepy.tech/badge/indoxGen-torch)](https://pepy.tech/project/indoxGen-torch)

[![Discord](https://img.shields.io/discord/1223867382460579961?label=Discord&logo=Discord&style=social)](https://discord.com/invite/ossllmai)
[![GitHub stars](https://img.shields.io/github/stars/osllmai/IndoxGen-torch?style=social)](https://github.com/osllmai/indoxGen)

<p align="center">
  <a href="https://osllm.ai">Official Website</a> &bull; <a href="https://docs.osllm.ai/index.html">Documentation</a> &bull; <a href="https://discord.gg/qrCc56ZR">Discord</a>
</p>

<p align="center">
  <b>NEW:</b> <a href="https://docs.google.com/forms/d/1CQXJvxLUqLBSXnjqQmRpOyZqD6nrKubLz2WTcIJ37fU/prefill">Subscribe to our mailing list</a> for updates and news!
</p>

## Overview

IndoxGen-Torch is a cutting-edge framework for generating high-quality synthetic data using Generative Adversarial Networks (GANs) powered by PyTorch. This module extends the capabilities of IndoxGen by providing a robust, PyTorch-based solution for creating realistic tabular data, particularly suited for complex datasets with mixed data types.

## Key Features

- **GAN-based Generation**: Utilizes advanced GAN architecture for high-fidelity synthetic data creation.
- **PyTorch Integration**: Built on PyTorch for efficient, GPU-accelerated training and generation.
- **Flexible Data Handling**: Supports categorical, mixed, and integer columns for versatile data modeling.
- **Customizable Architecture**: Easily configure generator and discriminator layers, learning rates, and other hyperparameters.
- **Training Monitoring**: Built-in patience-based early stopping for optimal model training.
- **Scalable Generation**: Efficiently generate large volumes of synthetic data post-training.

## Installation

```bash
pip install IndoxGen-Torch
```

## Quick Start Guide

### Basic Usage

```python
from indoxGen_pytorch import TabularGANConfig, TabularGANTrainer
import pandas as pd

# Load your data
data = pd.read_csv("data/Adult.csv")

# Define column types
categorical_columns = ["workclass", "education", "marital-status", "occupation",
                       "relationship", "race", "gender", "native-country", "income"]
mixed_columns = {"capital-gain": "positive", "capital-loss": "positive"}
integer_columns = ["age", "fnlwgt", "hours-per-week", "capital-gain", "capital-loss"]

# Set up the configuration
config = TabularGANConfig(
    input_dim=200,
    generator_layers=[128, 256, 512],
    discriminator_layers=[512, 256, 128],
    learning_rate=2e-4,
    beta_1=0.5,
    beta_2=0.9,
    batch_size=128,
    epochs=50,
    n_critic=5
)

# Initialize and train the model
trainer = TabularGANTrainer(
    config=config,
    categorical_columns=categorical_columns,
    mixed_columns=mixed_columns,
    integer_columns=integer_columns
)
history = trainer.train(data, patience=15)

# Generate synthetic data
synthetic_data = trainer.generate_samples(50000)
```

## Advanced Techniques

### Customizing the GAN Architecture

```python
custom_config = TabularGANConfig(
    input_dim=300,
    generator_layers=[256, 512, 1024, 512],
    discriminator_layers=[512, 1024, 512, 256],
    learning_rate=1e-4,
    batch_size=256,
    epochs=100,
    n_critic=3
)

custom_trainer = TabularGANTrainer(config=custom_config, ...)
```

### Handling Imbalanced Datasets

```python
original_class_distribution = data['target_column'].value_counts(normalize=True)
synthetic_data = trainer.generate_samples(100000)
synthetic_class_distribution = synthetic_data['target_column'].value_counts(normalize=True)
```

## Configuration and Customization

The `TabularGANConfig` class allows for extensive customization:

- `input_dim`: Dimension of the input noise vector
- `generator_layers` and `discriminator_layers`: List of layer sizes for the generator and discriminator
- `learning_rate`, `beta_1`, `beta_2`: Adam optimizer parameters
- `batch_size`, `epochs`: Training configuration
- `n_critic`: Number of discriminator updates per generator update

Refer to the API documentation for a comprehensive list of configuration options.

## Best Practices

1. **Data Preprocessing**: Ensure your data is properly cleaned and normalized before training.
2. **Hyperparameter Tuning**: Experiment with different configurations to find the optimal setup for your dataset.
3. **Validation**: Regularly compare the distribution of synthetic data with the original dataset.
4. **Privacy Considerations**: Implement differential privacy techniques when dealing with sensitive data.
5. **Scalability**: For large datasets, consider using distributed training capabilities of TensorFlow.

## Roadmap
* [x] Implement basic GAN architecture for tabular data
* [x] Add support for mixed data types (categorical, continuous, integer)
* [x] Integrate early stopping and training history
* [ ] Implement more advanced GAN variants (WGAN, CGAN)
* [ ] Add built-in privacy preserving mechanisms
* [ ] Develop automated hyperparameter tuning
* [ ] Create visualization tools for synthetic data quality assessment
* [ ] Implement distributed training support for large-scale datasets

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to get started.

## License

IndoxGen-Torch is released under the MIT License. See [LICENSE.md](LICENSE.md) for more details.

---
