import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import TabularGANConfig

class Discriminator(nn.Module):
    """
    Discriminator class for the GAN model, responsible for classifying real and generated data.

    Attributes:
    -----------
    config : TabularGANConfig
        Configuration object containing the parameters for the discriminator architecture.
    """

    def __init__(self, config: TabularGANConfig):
        """
        Initializes the discriminator model based on the configuration provided.

        Parameters:
        -----------
        config : TabularGANConfig
            Configuration object containing the parameters for the discriminator architecture.
        """
        super(Discriminator, self).__init__()
        self.config = config

        # BiLSTM Layer
        self.bilstm = nn.LSTM(
            input_size=self.config.output_dim,
            hidden_size=64,
            bidirectional=True,
            batch_first=True
        )

        # Fully connected layers based on the discriminator configuration
        self.fc_layers = self.build_fc_layers(self.config.discriminator_layers)

        # Output layer (WGAN-GP uses no activation here)
        self.output_layer = nn.Linear(self.config.discriminator_layers[-1], 1)

    def build_fc_layers(self, layer_sizes):
        """
        Builds the fully connected layers based on the configuration.

        Parameters:
        -----------
        layer_sizes: List of layer sizes for fully connected layers.

        Returns:
        --------
        nn.Sequential: Sequential layers of fully connected layers.
        """
        layers = []
        input_dim = 128  # Since BiLSTM is bidirectional with 64 hidden units, output dim is 2 * hidden_size = 128
        for units in layer_sizes:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.LayerNorm(units))
            layers.append(nn.Dropout(0.3))
            input_dim = units
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor, training: bool = False) -> torch.Tensor:
        """
        Forward pass through the discriminator.

        Parameters:
        -----------
        inputs : torch.Tensor
            A batch of input data (either real or generated) to classify.
        training : bool
            Whether the model is in training mode or not.

        Returns:
        --------
        torch.Tensor:
            A batch of predictions (real or fake) for each input sample.
        """
        # Reshape the input to add the sequence length dimension for LSTM
        inputs = inputs.unsqueeze(1)  # Shape: (batch_size, 1, input_dim)

        # Pass through BiLSTM
        lstm_out, _ = self.bilstm(inputs)  # Shape: (batch_size, 1, 128)

        # Remove the sequence dimension
        lstm_out = lstm_out.squeeze(1)  # Shape: (batch_size, 128)

        # Pass through fully connected layers
        x = self.fc_layers(lstm_out)

        # Pass through the output layer (linear activation for WGAN-GP)
        return self.output_layer(x)

    def gradient_penalty(self, real_samples: torch.Tensor, fake_samples: torch.Tensor,
                         device: torch.device) -> torch.Tensor:
        """
        Calculates the gradient penalty for WGAN-GP.

        Parameters:
        -----------
        real_samples : torch.Tensor
            A batch of real data samples.
        fake_samples : torch.Tensor
            A batch of generated data samples.

        Returns:
        --------
        torch.Tensor:
            The calculated gradient penalty.
        """
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, device=device).expand_as(real_samples)
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples

        interpolated = interpolated.requires_grad_(True)

        # Compute the predictions for interpolated data
        predictions = self(interpolated)

        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=predictions,
            inputs=interpolated,
            grad_outputs=torch.ones(predictions.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        slopes = torch.sqrt(torch.sum(gradients ** 2, dim=1))
        gradient_penalty = torch.mean((slopes - 1.0) ** 2)

        return gradient_penalty
