import torch
import torch.nn as nn
from .config import TabularGANConfig
import torch
import torch.nn as nn
# from libs.indoxGen_torch.GAN.config import TabularGANConfig

class Generator(nn.Module):
    """
    Generator class for the GAN model, which takes in random noise and generates synthetic tabular data.

    Attributes:
    -----------
    config : TabularGANConfig
        Configuration object containing the parameters for the generator architecture.
    """

    def __init__(self, config: TabularGANConfig):
        """
        Initializes the generator model based on the configuration provided.

        Parameters:
        -----------
        config : TabularGANConfig
            Configuration object containing the parameters for the generator architecture.
        """
        super(Generator, self).__init__()
        self.config = config
        self.model = self.build_model()

    def build_model(self):
        """
        Builds the generator model based on the configuration, including a Bidirectional LSTM layer.

        Returns:
        --------
        nn.Sequential:
            A PyTorch Sequential model representing the generator architecture.
        """
        layers = []

        # Input to BiLSTM Layer
        self.bilstm = nn.LSTM(
            input_size=self.config.input_dim,
            hidden_size=64,
            bidirectional=True,
            batch_first=True
        )

        input_dim = 128  # BiLSTM outputs (2 * hidden_size = 128)

        # Fully connected layers after BiLSTM
        for i in range(len(self.config.generator_layers)):
            layers.append(nn.Linear(input_dim, self.config.generator_layers[i]))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.BatchNorm1d(self.config.generator_layers[i]))
            input_dim = self.config.generator_layers[i]

            # Add residual connections for deeper networks
            if i > 1 and self.config.generator_layers[i] == self.config.generator_layers[i - 2]:
                layers.append(Residual(self.config.generator_layers[i]))

        # Final output layer
        layers.append(nn.Linear(input_dim, self.config.output_dim))
        layers.append(nn.Tanh())

        return nn.Sequential(*layers)

    def forward(self, inputs):
        """
        Forward pass through the generator.

        Parameters:
        -----------
        inputs : torch.Tensor
            A batch of input noise vectors to generate synthetic data from.

        Returns:
        --------
        torch.Tensor:
            A batch of generated data.
        """
        # Reshape for BiLSTM
        inputs = inputs.unsqueeze(1)  # Shape: (batch_size, 1, input_dim)

        # Pass through BiLSTM
        lstm_out, _ = self.bilstm(inputs)  # Shape: (batch_size, 1, 128)

        # Remove sequence dimension after LSTM
        lstm_out = lstm_out.squeeze(1)  # Shape: (batch_size, 128)

        # Pass through fully connected layers
        return self.model(lstm_out)

    def generate(self, num_samples: int) -> torch.Tensor:
        """
        Generates a specified number of synthetic samples.

        Parameters:
        -----------
        num_samples : int
            The number of synthetic samples to generate.

        Returns:
        --------
        torch.Tensor:
            A tensor of generated synthetic samples.
        """
        noise = torch.randn(num_samples, self.config.input_dim, device=next(self.model.parameters()).device)
        return self(noise)

class Residual(nn.Module):
    """
    Residual connection layer for deeper networks.

    Attributes:
    -----------
    input_dim : int
        The dimension of the input/output for the residual connection.
    """

    def __init__(self, input_dim: int):
        super(Residual, self).__init__()
        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        """
        Forward pass through the residual connection.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor.

        Returns:
        --------
        torch.Tensor:
            The output tensor with the residual connection applied.
        """
        return x + self.fc(x)

