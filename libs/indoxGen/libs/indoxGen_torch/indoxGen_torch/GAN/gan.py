import torch
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
                 mixed_columns: dict = None, integer_columns: list = None, device='cpu'):
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
        device: str, optional
            Device for training (default is 'cpu').
        """
        self.config = config
        self.categorical_columns = categorical_columns
        self.mixed_columns = mixed_columns
        self.integer_columns = integer_columns
        self.transformer = None
        self.gan = None
        self.history = None
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

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
        self.gan = TabularGAN(self.config).to(self.device)

        # Optimizer with gradient clipping
        g_optimizer = torch.optim.Adam(self.gan.generator.parameters(),
                                       lr=self.config.learning_rate,
                                       betas=(self.config.beta_1, self.config.beta_2))
        d_optimizer = torch.optim.Adam(self.gan.discriminator.parameters(),
                                       lr=self.config.learning_rate,
                                       betas=(self.config.beta_1, self.config.beta_2))

        self.gan.set_optimizers(g_optimizer, d_optimizer)

    def train(self, data: pd.DataFrame, patience: int = 10, verbose: int = 1):
      transformed_data = self.prepare_data(data)
      transformed_data = torch.tensor(transformed_data, dtype=torch.float32).to(self.device)
      self.compile_gan()

      if not self.gan:
          raise ValueError("GAN model is not initialized or compiled correctly.")

      gan_monitor = GANMonitor(patience=patience)

      # Initialize the DataLoader for batching
      dataset = torch.utils.data.TensorDataset(transformed_data)
      dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

      for epoch in range(self.config.epochs):
          epoch_g_loss = 0.0
          epoch_d_loss = 0.0

          for batch_data in dataloader:
              real_data = batch_data[0].to(self.device)

              # Perform one training step (alternating between generator and discriminator)
              losses = self.gan.train_step(real_data)

              # Accumulate losses for reporting
              epoch_d_loss += losses['d_loss']
              epoch_g_loss += losses['g_loss']

          # Calculate average losses for the epoch
          avg_d_loss = epoch_d_loss / len(dataloader)
          avg_g_loss = epoch_g_loss / len(dataloader)

          if verbose:
              print(f"Epoch [{epoch+1}/{self.config.epochs}] - D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")

          # Early stopping monitor
          gan_monitor.on_epoch_end(epoch, avg_g_loss)

          if gan_monitor.early_stop:
              print(f"Training stopped early at epoch {epoch+1} due to no improvement in generator loss.")
              break

      return {"d_loss": avg_d_loss, "g_loss": avg_g_loss}


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

      # No need for detach().cpu() here since it's already in numpy format
      return self.transformer.inverse_transform(generated_data)


    def get_training_history(self):
        """
        Returns the training history.

        Returns:
        --------
        List of training logs.
        """
        return self.history

