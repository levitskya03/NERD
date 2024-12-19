
import torch
from src.utils.metrics import l2_distortion

PAPER_EXPERIMENTS = [
    {
        'batch_size': 64,
        'latent_dim': 100,
        'dim': 64,
        'img_size': (32, 32, 1),  # Resized images as in the paper
        'learning_rate': 0.0002,
        'steps': 10,
        'distortion_fn': l2_distortion,  # L2 distortion used in experiments
        'dataset': 'MNIST',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'beta': 0.1,  # Initial beta value from the paper
        'd': 54.95  # Distortion threshold for MNIST
    },
    {
        'batch_size': 64,
        'latent_dim': 100,
        'dim': 64,
        'img_size': (32, 32, 1),  # Resized images as in the paper
        'learning_rate': 0.0002,
        'steps': 10,
        'distortion_fn': l2_distortion,
        'dataset': 'FashionMNIST',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'beta': 0.1,  # Initial beta value
        'd': 54.95  # Using the same as MNIST, adjust if needed
    },
    {
        'batch_size': 64,
        'latent_dim': 100,
        'dim': 64,
        'img_size': (32, 32, 1),  # Resized images as in the paper
        'learning_rate': 0.0002,
        'steps': 10,
        'distortion_fn': l2_distortion,
        'dataset': 'Gaussian',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'beta': 0.1,
        'gaussian_m': 1024,  # Gaussian dimension
        'gaussian_r': 0.025,  # Variance scale
        'd': 30.0  # Distortion value based on synthetic data
    }
]
