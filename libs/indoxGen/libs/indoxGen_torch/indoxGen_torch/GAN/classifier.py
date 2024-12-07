import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import TabularGANConfig
class Classifier(nn.Module):
    """
    Classifier class for the GAN model, designed to classify generated tabular data into multiple classes.
    This implementation includes a Bidirectional LSTM layer for sequence modeling.

    Attributes:
    -----------
    config : TabularGANConfig
        Configuration object containing the parameters for the classifier architecture.
    num_classes : int
        The number of output classes for classification.
    """

    def __init__(self, config: TabularGANConfig, num_classes: int):
        """
        Initializes the classifier model based on the configuration and the number of output classes.

        Parameters:
        -----------
        config : TabularGANConfig
            Configuration object containing the parameters for the classifier architecture.
        num_classes : int
            The number of classes for the classification task.
        """
        super(Classifier, self).__init__()
        self.config = config
        self.num_classes = num_classes

        # Bidirectional LSTM layer
        self.bilstm = nn.LSTM(
            input_size=self.config.output_dim,
            hidden_size=64,
            bidirectional=True,
            batch_first=True
        )

        # Fully connected layers from the discriminator configuration
        self.fc_layers = self.build_fc_layers(self.config.discriminator_layers)

        # Output layer
        self.output_layer = nn.Linear(self.config.discriminator_layers[-1], self.num_classes)

    def build_fc_layers(self, layer_sizes):
        """
        Builds the fully connected layers based on the configuration.
        """
        layers = []
        input_dim = 128  # Since BiLSTM is bidirectional with 64 hidden units, output dim is 2 * hidden_size = 128
        for units in layer_sizes:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
            layers.append(nn.LayerNorm(units))
            layers.append(nn.Dropout(0.3))
            input_dim = units
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier.

        Parameters:
        -----------
        inputs : torch.Tensor
            A batch of input data to classify.

        Returns:
        --------
        torch.Tensor:
            A batch of class probabilities for each input sample.
        """
        # Reshape the input to add the sequence length dimension for LSTM
        inputs = inputs.unsqueeze(1)  # Shape: (batch_size, 1, input_dim)

        # Pass through BiLSTM
        lstm_out, _ = self.bilstm(inputs)  # Shape: (batch_size, 1, 128)

        # Remove the sequence dimension
        lstm_out = lstm_out.squeeze(1)  # Shape: (batch_size, 128)

        # Pass through fully connected layers
        x = self.fc_layers(lstm_out)

        # Pass through the output layer
        return F.softmax(self.output_layer(x), dim=1)
