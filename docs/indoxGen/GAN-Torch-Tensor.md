# IndoxGen-Torch & IndoxGen-Tensor: GAN-Based Synthetic Data Generation Framework

## Overview

**IndoxGen-Torch** and **IndoxGen-Tensor** are two advanced frameworks designed for generating synthetic tabular data using Generative Adversarial Networks (GANs). IndoxGen-Torch is based on PyTorch, while IndoxGen-Tensor is based on TensorFlow, providing flexibility depending on user preference. Both frameworks support various data types, including categorical, continuous, and integer data.

These tools extend the capabilities of the IndoxGen project by offering easy-to-use configurations, efficient training pipelines, scalable synthetic data generation, and evaluation methods to assess the quality of the generated data.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [IndoxGen-Torch Example](#indoxgen-torch-example)
  - [IndoxGen-Tensor Example](#indoxgen-tensor-example)
- [Configuration](#configuration)
- [Evaluation Methods](#evaluation-methods)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)


---

## Installation

To install **IndoxGen-Torch** or **IndoxGen-Tensor**, you need Python 3.9+.

- For **IndoxGen-Torch**:
    pip install indoxgen-torch

- For **IndoxGen-Tensor**:
    pip install indoxgen-tensor

Both libraries require dependencies like PyTorch or TensorFlow, depending on the version you're using. Ensure your environment supports GPU for faster model training.

---

## Usage

### IndoxGen-Torch Example

```python
from indoxGen_torch import TabularGANConfig, TabularGANTrainer
import pandas as pd

# Load your dataset
data = pd.read_csv("data/Adult.csv")

# Define column types
categorical_columns = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "gender", "native-country", "income"]
mixed_columns = {"capital-gain": "positive", "capital-loss": "positive"}
integer_columns = ["age", "fnlwgt", "hours-per-week", "capital-gain", "capital-loss"]

# Set up the configuration
config = TabularGANConfig(
    input_dim=200,
    generator_layers=[128, 256, 512],
    discriminator_layers=[512, 256, 128],
    learning_rate=2e-4,
    batch_size=128,
    epochs=50,
    n_critic=5
)

# Initialize the trainer
trainer = TabularGANTrainer(
    config=config,
    categorical_columns=categorical_columns,
    mixed_columns=mixed_columns,
    integer_columns=integer_columns
)

# Train the model
history = trainer.train(data, patience=15)

# Generate synthetic data
synthetic_data = trainer.generate_samples(50000)
print(synthetic_data.head())
```

### IndoxGen-Tensor Example

```python
from indoxGen_tensor import TabularGANConfig, TabularGANTrainer
import pandas as pd

# Load your dataset
data = pd.read_csv("data/Adult.csv")

# Define column types
categorical_columns = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "gender", "native-country", "income"]
mixed_columns = {"capital-gain": "positive", "capital-loss": "positive"}
integer_columns = ["age", "fnlwgt", "hours-per-week", "capital-gain", "capital-loss"]

# Set up the configuration
config = TabularGANConfig(
    input_dim=200,
    generator_layers=[128, 256, 512],
    discriminator_layers=[512, 256, 128],
    learning_rate=2e-4,
    batch_size=128,
    epochs=50,
    n_critic=5
)

# Initialize the trainer
trainer = TabularGANTrainer(
    config=config,
    categorical_columns=categorical_columns,
    mixed_columns=mixed_columns,
    integer_columns=integer_columns
)

# Train the model
history = trainer.train(data, patience=15)

# Generate synthetic data
synthetic_data = trainer.generate_samples(50000)
print(synthetic_data.head())
```

---

## Configuration

The **TabularGANConfig** class provides extensive customization options to adapt the model architecture and training process to your dataset.

### Key Parameters:
- input_dim: Dimension of the input noise vector for the generator.
- generator_layers: List of neuron counts in each layer of the generator.
- discriminator_layers: List of neuron counts in each layer of the discriminator.
- learning_rate: The learning rate used in the Adam optimizer.
- batch_size: Number of samples in each batch during training.
- epochs: Number of full passes over the dataset during training.
- n_critic: Number of discriminator updates per generator update, used in WGAN-GP.

These parameters can be modified when initializing the TabularGANConfig.

---

## Evaluation Methods

To ensure the quality of the synthetic data generated by **IndoxGen-Torch** and **IndoxGen-Tensor**, we provide several evaluation methods:

### 1. Utility Evaluation
Utility evaluation compares the performance of machine learning classifiers trained on real data versus synthetic data. We assess the accuracy, AUC (Area Under Curve), and F1 score of various classifiers, including:
- Logistic Regression
- Decision Tree
- Random Forest
- Multi-Layer Perceptron (MLP)

This helps determine how well synthetic data can replicate the utility of real data in predictive models.

### 2. Statistical Similarity
Statistical similarity evaluates how closely the synthetic data mirrors the real data. We use metrics like:
- Wasserstein Distance (for continuous columns)
- Jensen-Shannon Divergence (for categorical columns)
- Correlation distance (between real and synthetic data correlation matrices)

These metrics give insight into how well the synthetic data captures the underlying statistical properties of the real dataset.

### 3. Privacy Evaluation
We evaluate privacy by measuring:
- **Distance to Closest Record (DCR)**: The minimum distance between each real and synthetic data point.
- **Nearest Neighbor Distance Ratio (NNDR)**: The ratio of distances between real and synthetic points to ensure diversity and privacy.

These metrics help ensure that synthetic data doesn't reveal sensitive information about real data records.

### 4. Data Drift Evaluation
Data drift is assessed by comparing the distributions of real and synthetic data using:
- **Population Stability Index (PSI)** for categorical features.
- **Kolmogorov-Smirnov (K-S) test** for numerical features.

This ensures that the synthetic data maintains the same distributional properties as the real data, indicating no significant drift.

### 5. Visualization of Distributions
We provide tools to visualize the distributions of real and synthetic data side-by-side, allowing for easy comparison. This helps validate that the synthetic data replicates key patterns from the real data, especially for critical variables.

---

## API Reference

### TabularGANConfig

#### **`__init__`** parameters:
- input_dim: The size of the random noise vector.
- generator_layers: A list of integers specifying the number of neurons in each generator layer.
- discriminator_layers: A list of integers specifying the number of neurons in each discriminator layer.
- learning_rate: The learning rate for both the generator and discriminator (default: 0.0002).
- batch_size: The batch size for training (default: 128).
- epochs: The number of epochs for training (default: 50).
- n_critic: The number of discriminator updates per generator update (default: 5).

---

## Contributing

Contributions to **IndoxGen-Torch** and **IndoxGen-Tensor** are welcome. Follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Add your changes and write tests if applicable.
4. Submit a pull request with a clear description of your changes.

We encourage contributions that improve the GAN models, add more data handling features, or extend the functionality of the framework.

---

## License

**IndoxGen-Torch** and **IndoxGen-Tensor** are released under the MIT License. See [LICENSE.md](LICENSE.md) for details.

---