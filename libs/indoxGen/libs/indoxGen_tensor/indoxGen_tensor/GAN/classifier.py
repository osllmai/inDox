from tensorflow import keras
import tensorflow as tf

from .config import TabularGANConfig

class Classifier(keras.Model):
    """
    Classifier class for the GAN model, designed to classify generated tabular data into multiple classes.

    Attributes:
    -----------
    config : TabularGANConfig
        Configuration object containing the parameters for the classifier architecture.
    num_classes : int
        The number of output classes for classification.
    model : keras.Model
        The actual Keras model built using the specified layers in the configuration.
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
        self.model = self.build_model()

    def build_model(self):
        """
        Builds the discriminator model based on the configuration, including a Bidirectional LSTM layer.

        Returns:
        --------
        tf.keras.Model:
            A Keras Model representing the discriminator architecture with Bidirectional LSTM.
        """
        input_layer = tf.keras.layers.Input(shape=(self.config.output_dim,))

        # Reshape input for BiLSTM layer
        x = tf.keras.layers.Reshape((1, self.config.output_dim))(input_layer)

        # Add Bidirectional LSTM layer
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=64, return_sequences=False)
        )(x)

        for units in self.config.discriminator_layers:
            x = tf.keras.layers.Dense(units, kernel_initializer='he_normal')(x)
            x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.Dropout(0.3)(x)

        # Use linear activation for WGAN-GP
        output_layer = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

        return model

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass through the classifier.

        Parameters:
        -----------
        inputs : tf.Tensor
            A batch of input data to classify.

        Returns:
        --------
        tf.Tensor:
            A batch of class probabilities for each input sample.
        """
        return self.model(inputs)
