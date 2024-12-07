from tensorflow import keras
import tensorflow as tf

from .config import TabularGANConfig
from .discriminator import Discriminator
from .generator import Generator

class TabularGAN(keras.Model):
    """
    TabularGAN model that includes both the generator and discriminator for generating tabular data
    and training them using WGAN-GP (Wasserstein GAN with Gradient Penalty).
    """

    def __init__(self, config: TabularGANConfig):
        """
        Initializes the TabularGAN model with the generator and discriminator.
        """
        super(TabularGAN, self).__init__()
        self.config = config
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)

    def compile(self, g_optimizer, d_optimizer, g_lr_scheduler=None, d_lr_scheduler=None):
        """
        Sets the optimizers and learning rate schedulers for both the generator and discriminator.
        """
        super(TabularGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_lr_scheduler = g_lr_scheduler
        self.d_lr_scheduler = d_lr_scheduler
    def gradient_penalty(self, real_data, fake_data):
        """
        Computes the gradient penalty for the WGAN-GP, enforcing Lipschitz constraint.
        """
        epsilon = tf.random.uniform([tf.shape(real_data)[0], 1], 0.0, 1.0)
        interpolated_data = epsilon * real_data + (1 - epsilon) * fake_data
        with tf.GradientTape() as tape:
            tape.watch(interpolated_data)
            interpolated_output = self.discriminator(interpolated_data, training=True)
        gradients = tape.gradient(interpolated_output, [interpolated_data])[0]
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
        return tf.reduce_mean((gradients_norm - 1.0) ** 2)

    @tf.function
    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]
        noise = tf.random.normal([batch_size, self.config.input_dim])  # Can experiment with uniform noise

        # Train discriminator
        for _ in range(self.config.n_critic):
            with tf.GradientTape() as tape:
                fake_data = self.generator(noise, training=True)
                real_output = self.discriminator(real_data, training=True)
                fake_output = self.discriminator(fake_data, training=True)

                d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
                gp = self.gradient_penalty(real_data, fake_data)
                d_loss += 10.0 * gp

            d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

            if self.d_lr_scheduler:
                self.d_lr_scheduler.on_batch_end(batch=batch_size, logs={"loss": d_loss})

        # Train generator
        with tf.GradientTape() as tape:
            fake_data = self.generator(noise, training=True)
            fake_output = self.discriminator(fake_data, training=True)
            g_loss = -tf.reduce_mean(fake_output)

        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        if self.g_lr_scheduler:
            self.g_lr_scheduler.on_batch_end(batch=batch_size, logs={"loss": g_loss})

        return {"d_loss": d_loss, "g_loss": g_loss}

    def generate(self, num_samples):
        """
        Generates synthetic data using the trained generator.

        Parameters:
        -----------
        num_samples : int
            The number of samples to generate.

        Returns:
        --------
        np.ndarray:
            Generated synthetic data.
        """
        noise = tf.random.normal([num_samples, self.config.input_dim])
        return self.generator(noise).numpy()
