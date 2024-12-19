import torch
from src.utils.metrics import l2_distortion

PAPER_EXPERIMENTS = [
    {
        'batch_size': 64,
        'latent_dim': 100,
        'dim': 64,
        'img_size': (28, 28, 1),
        'learning_rate': 0.0002,
        'steps': 10,
        'distortion_fn': l2_distortion,
        'dataset': 'MNIST',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'beta': 0.1
    },
    {
        'batch_size': 64,
        'latent_dim': 100,
        'dim': 64,
        'img_size': (28, 28, 1),
        'learning_rate': 0.0002,
        'steps': 10,
        'distortion_fn': l2_distortion,
        'dataset': 'FashionMNIST',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'beta': 0.1
    },
    {
        'batch_size': 64,
        'latent_dim': 100,
        'dim': 64,
        'img_size': (28, 28, 1),
        'learning_rate': 0.0002,
        'steps': 10,
        'distortion_fn': l2_distortion,
        'dataset': 'Gaussian',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'beta': 0.1,
        'gaussian_m': 1024,
        'gaussian_r': 0.025
    }
]
