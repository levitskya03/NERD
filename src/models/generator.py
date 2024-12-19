import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Generator model G_θ for mapping latent space Z to output space Y.
    """
    def __init__(self, img_size, latent_dim, dim):
        """
        Args:
            img_size (tuple): Size of the output image (Height, Width, Channels).
            latent_dim (int): Dimensionality of the latent space Z.
            dim (int): Base dimensionality for convolutional layers.
        """
        super(Generator, self).__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.feature_sizes = (int(self.img_size[0] / 16), int(self.img_size[1] / 16))

        # Map latent space to intermediate feature map
        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim, 8 * dim * self.feature_sizes[0] * self.feature_sizes[1]),
            nn.ReLU()
        )

        # Map feature map to image space
        self.features_to_image = nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(4 * dim, track_running_stats=True),

            nn.ConvTranspose2d(4 * dim, 2 * dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(2 * dim, track_running_stats=True),

            nn.ConvTranspose2d(2 * dim, dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(dim, track_running_stats=True),

            nn.ConvTranspose2d(dim, self.img_size[2], kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        """
        Forward pass through the generator.
        Args:
            z (Tensor): Latent space input of shape (batch_size, latent_dim).
        Returns:
            Tensor: Generated output of shape (batch_size, img_size[0], img_size[1], img_size[2]).
        """
        # Map latent vector to feature map
        features = self.latent_to_features(z)
        features = features.view(-1, 8 * self.dim, self.feature_sizes[0], self.feature_sizes[1])

        # Map feature map to image
        output = self.features_to_image(features)
        return output
