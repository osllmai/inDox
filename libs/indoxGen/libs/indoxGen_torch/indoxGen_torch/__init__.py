from .GAN.gan import TabularGANTrainer
from .GAN.config import TabularGANConfig
from .GAN.evaluation import (
    train_and_evaluate_classifier,
    evaluate_utility,
    evaluate_statistical_similarity,
    evaluate_privacy,
    plot_distributions,
    evaluate_data_drift
)

import importlib.metadata

# __version__ = importlib.metadata.version("indoxGen_torch")
