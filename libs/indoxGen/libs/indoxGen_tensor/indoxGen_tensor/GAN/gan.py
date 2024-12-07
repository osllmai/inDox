from tensorflow import keras
import pandas as pd
import numpy as np
from .config import TabularGANConfig
from .data_transformer import DataTransformer
from .tabular_gan import TabularGAN
from .utils import GANMonitor


class TabularGANTrainer:
    """
    TabularGANTrainer class for training a Tabular GAN model on a provided dataset.
    This class handles data preprocessing, model creation, training, and result generation.
    """

    def __init__(self, config: TabularGANConfig, categorical_columns: list = None,
                 mixed_columns: dict = None, integer_columns: list = None):
        """
        Initializes the TabularGANTrainer with the necessary configuration and columns.

        Parameters:
        -----------
        config : TabularGANConfig
            Configuration object containing the parameters for the GAN architecture.
        categorical_columns : list, optional
            List of categorical columns to one-hot encode.
        mixed_columns : dict, optional
            Dictionary specifying constraints on mixed columns (e.g., 'positive', 'negative').
        integer_columns : list, optional
            List of integer columns for rounding during inverse transformation.
        """
        self.config = config
        self.categorical_columns = categorical_columns
        self.mixed_columns = mixed_columns
        self.integer_columns = integer_columns
        self.transformer = None
        self.gan = None
        self.history = None

    def prepare_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepares the data by fitting the transformers and transforming the input data.

        Parameters:
        -----------
        data : pd.DataFrame
            The raw tabular data to be processed.

        Returns:
        --------
        np.ndarray:
            Transformed data ready for GAN training.
        """
        self.transformer = DataTransformer(
            categorical_columns=self.categorical_columns,
            mixed_columns=self.mixed_columns,
            integer_columns=self.integer_columns
        )
        self.transformer.fit(data)
        transformed_data = self.transformer.transform(data)

        # Update the output_dim in the config based on the transformed data
        self.config.output_dim = transformed_data.shape[1]

        return transformed_data

    def compile_gan(self):
        self.gan = TabularGAN(self.config)

        # Optimizer with gradient clipping
        g_optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate,
                                            beta_1=self.config.beta_1,
                                            beta_2=self.config.beta_2,
                                            clipvalue=1.0)
        d_optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate,
                                            beta_1=self.config.beta_1,
                                            beta_2=self.config.beta_2,
                                            clipvalue=1.0)

        # Compile the GAN
        self.gan.compile(g_optimizer=g_optimizer, d_optimizer=d_optimizer)

    def train(self, data: pd.DataFrame, patience: int = 10, verbose: int = 1):
        transformed_data = self.prepare_data(data)
        self.compile_gan()

        if not self.gan:
            raise ValueError("GAN model is not initialized or compiled correctly.")

        gan_monitor = GANMonitor(patience=patience)

        # Add learning rate scheduler
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='d_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

        self.history = self.gan.fit(
            transformed_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=[gan_monitor, lr_scheduler],
            verbose=verbose
        )

        return self.history

    def generate_samples(self, num_samples: int) -> pd.DataFrame:
        """
        Generates new samples using the trained GAN model and converts them back to the original format.

        Parameters:
        -----------
        num_samples : int
            The number of samples to generate.

        Returns:
        --------
        pd.DataFrame:
            Generated synthetic data in its original format.
        """
        if not self.gan or not self.transformer:
            raise ValueError("GAN model is not trained yet. Call `train` method first.")

        generated_data = self.gan.generate(num_samples)
        return self.transformer.inverse_transform(generated_data)

    def get_training_history(self) -> keras.callbacks.History:
        """
        Returns the training history.

        Returns:
        --------
        keras.callbacks.History:
            The history object containing training logs.
        """
        return self.history
