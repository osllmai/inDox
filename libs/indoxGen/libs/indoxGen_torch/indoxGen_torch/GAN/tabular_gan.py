import torch
import torch.nn as nn
import torch.optim as optim
from .config import TabularGANConfig
from .generator import Generator
from .discriminator import Discriminator

class TabularGAN(nn.Module):
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

    def set_optimizers(self, g_optimizer, d_optimizer):
        """
        Sets the optimizers for both the generator and discriminator.
        """
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    def gradient_penalty(self, real_data, fake_data, device):
        """
        Computes the gradient penalty for the WGAN-GP, enforcing the Lipschitz constraint.
        """
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, device=device)
        epsilon = epsilon.expand_as(real_data)
        interpolated_data = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated_data.requires_grad_(True)

        interpolated_output = self.discriminator(interpolated_data)
        gradients = torch.autograd.grad(
            outputs=interpolated_output,
            inputs=interpolated_data,
            grad_outputs=torch.ones(interpolated_output.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1))
        gradient_penalty = torch.mean((gradients_norm - 1.0) ** 2)
        return gradient_penalty

    def train_step(self, real_data):
        batch_size = real_data.size(0)
        device = real_data.device

        # Generate noise
        noise = torch.randn(batch_size, self.config.input_dim, device=device)

        # Train discriminator
        for _ in range(self.config.n_critic):
            fake_data = self.generator(noise)
            real_output = self.discriminator(real_data)
            fake_output = self.discriminator(fake_data)

            d_loss = torch.mean(fake_output) - torch.mean(real_output)
            gp = self.gradient_penalty(real_data, fake_data, device)
            d_loss += 10.0 * gp

            # Backpropagate for the discriminator
            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()

        # Train generator
        noise = torch.randn(batch_size, self.config.input_dim, device=device)
        fake_data = self.generator(noise)
        fake_output = self.discriminator(fake_data)
        g_loss = -torch.mean(fake_output)

        # Backpropagate for the generator
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return {"d_loss": d_loss.item(), "g_loss": g_loss.item()}

    def generate(self, num_samples):
        """
        Generates synthetic data using the trained generator.

        Parameters:
        -----------
        num_samples : int
            The number of samples to generate.

        Returns:
        --------
        torch.Tensor:
            Generated synthetic data.
        """
        device = next(self.parameters()).device
        noise = torch.randn(num_samples, self.config.input_dim, device=device)
        generated_data = self.generator(noise)
        return generated_data.detach().cpu().numpy()

