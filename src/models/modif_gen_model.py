import torch.nn as nn

class ModifiedGenerator(nn.Module):
    def __init__(self, img_size, latent_dim, dim):
        super(ModifiedGenerator, self).__init__()

        self.img_size = img_size
        self.latent_dim = latent_dim
        self.dim = dim

        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim, 8 * dim * (img_size[0] // 16) * (img_size[1] // 16)),
            nn.BatchNorm1d(8 * dim * (img_size[0] // 16) * (img_size[1] // 16)),
            nn.ReLU()
        )

        self.features_to_image = nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4 * dim),
            nn.ReLU(),
            nn.ConvTranspose2d(4 * dim, 2 * dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * dim),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.ConvTranspose2d(dim, img_size[2], kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.latent_to_features(z)
        x = x.view(-1, 8 * self.dim, self.img_size[0] // 16, self.img_size[1] // 16)
        return self.features_to_image(x)


class ModifiedDiscriminator(nn.Module):
    def __init__(self, img_size, dim):
        super(ModifiedDiscriminator, self).__init__()

        self.image_to_features = nn.Sequential(
            nn.Conv2d(img_size[2], dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, 2 * dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * dim, 4 * dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4 * dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4 * dim, 1, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, x):
        return self.image_to_features(x).view(-1, 1).squeeze(1)
