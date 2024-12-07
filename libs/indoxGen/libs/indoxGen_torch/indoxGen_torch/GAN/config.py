from typing import List

class TabularGANConfig:
    """
    Configuration class for setting up the parameters of a Tabular GAN model.

    Attributes:
    -----------
    input_dim : int
        The dimension of the input noise vector for the generator.
    generator_layers : List[int]
        A list of integers representing the number of neurons in each layer of the generator.
    discriminator_layers : List[int]
        A list of integers representing the number of neurons in each layer of the discriminator.
    learning_rate : float
        The learning rate for both the generator and discriminator.
    beta_1 : float
        The exponential decay rate for the first moment estimates (used in Adam optimizer).
    beta_2 : float
        The exponential decay rate for the second moment estimates (used in Adam optimizer).
    batch_size : int
        The number of samples per gradient update.
    epochs : int
        Number of epochs to train the model.
    n_critic : int
        The number of updates for the discriminator per update of the generator.
    """

    def __init__(self,
                 input_dim: int = 100,
                 generator_layers: List[int] = [256, 512, 256],
                 discriminator_layers: List[int] = [256, 512, 256],
                 learning_rate: float = 0.0002,
                 beta_1: float = 0.5,
                 beta_2: float = 0.9,
                 batch_size: int = 128,
                 epochs: int = 300,
                 n_critic: int = 5):
        """
        Initializes the GAN configuration with the necessary hyperparameters.

        Parameters:
        -----------
        input_dim : int, optional
            Dimension of the input noise vector for the generator. Default is 100.
        generator_layers : List[int], optional
            Layer sizes for the generator. Default is [256, 512, 256].
        discriminator_layers : List[int], optional
            Layer sizes for the discriminator. Default is [256, 512, 256].
        learning_rate : float, optional
            Learning rate for the optimizer. Default is 0.0002.
        beta_1 : float, optional
            Beta 1 for Adam optimizer. Default is 0.5.
        beta_2 : float, optional
            Beta 2 for Adam optimizer. Default is 0.9.
        batch_size : int, optional
            Batch size used for training. Default is 128.
        epochs : int, optional
            Number of training epochs. Default is 300.
        n_critic : int, optional
            Number of discriminator updates per generator update (for WGAN-GP). Default is 5.
        """
        self.input_dim = input_dim
        self.generator_layers = generator_layers
        self.discriminator_layers = discriminator_layers
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_critic = n_critic
        self.output_dim = None  # This will be set automatically in the trainer class

        self._validate_config()

    def _validate_config(self):
        """
        Validates the configuration parameters to ensure they are valid for GAN training.
        Raises:
        -------
        ValueError:
            If any of the critical configurations are invalid.
        """
        if not isinstance(self.input_dim, int) or self.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        if not all(isinstance(layer, int) and layer > 0 for layer in self.generator_layers):
            raise ValueError("All generator layers must be positive integers.")
        if not all(isinstance(layer, int) and layer > 0 for layer in self.discriminator_layers):
            raise ValueError("All discriminator layers must be positive integers.")
        if self.learning_rate <= 0 or self.beta_1 <= 0 or self.beta_2 <= 0:
            raise ValueError("learning_rate, beta_1, and beta_2 must be positive.")
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError("epochs must be a positive integer.")
        if not isinstance(self.n_critic, int) or self.n_critic <= 0:
            raise ValueError("n_critic must be a positive integer.")