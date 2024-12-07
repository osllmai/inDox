import tensorflow as tf

from .config import TabularGANConfig


class Generator(tf.keras.Model):
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
        tf.keras.Model:
            A Keras Model representing the generator architecture with Bidirectional LSTM.
        """
        input_layer = tf.keras.layers.Input(shape=(self.config.input_dim,))

        x = tf.keras.layers.Reshape((1, self.config.input_dim))(input_layer)

        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=64, return_sequences=False)
        )(x)

        for i, units in enumerate(self.config.generator_layers):
            x = tf.keras.layers.Dense(units, kernel_initializer='he_normal')(x)
            x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)
            x = tf.keras.layers.BatchNormalization()(x)

            if i > 0 and i % 2 == 0 and units == self.config.generator_layers[i - 2]:
                x = tf.keras.layers.Add()([x, tf.keras.layers.Dense(units)(x)])

        output_layer = tf.keras.layers.Dense(self.config.output_dim, activation='tanh')(x)

        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

        return model

    def call(self, inputs, training=False):
        """
        Forward pass through the generator.

        Parameters:
        -----------
        inputs : tf.Tensor
            A batch of input noise vectors to generate synthetic data from.
        training : bool
            Whether the model is in training mode or not.

        Returns:
        --------
        tf.Tensor:
            A batch of generated data.
        """
        return self.model(inputs, training=training)

    @tf.function
    def generate(self, num_samples: int):
        """
        Generates a specified number of synthetic samples.

        Parameters:
        -----------
        num_samples : int
            The number of synthetic samples to generate.

        Returns:
        --------
        tf.Tensor:
            A tensor of generated synthetic samples.
        """
        noise = tf.random.normal([num_samples, self.config.input_dim])
        return self(noise, training=False)
