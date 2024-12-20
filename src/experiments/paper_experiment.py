
import torch
from src.utils.metrics import l2_distortion
import numpy as np

PAPER_EXPERIMENTS = [
    {
        'batch_size': 64,
        'latent_dim': 100,
        'dim': 64,
        'img_size': (32, 32, 1),  
        'learning_rate': 0.0002,
        'steps': 10,
        'distortion_fn': l2_distortion,  
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
        'gaussian_m': 1024,  
        'gaussian_r': 0.025, 
        'd': 30.0 
    }
]

PLOT_EXPERIMENTS = [
    {'dataset': 'Gaussian', 
     'batch_size': 64, 
     'img_size': (32, 32, 1),
     'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),  
     'steps': 10, 
     'dim': 64,
     'learning_rate': 0.0002, 
     'distortion_fn': l2_distortion, 
     'latent_dim': 100, 'd_range': np.linspace(5, 50, 10)},
    {'dataset': 'FashionMNIST', 
     'batch_size': 64, 
     'img_size': (32, 32, 1),  
     'steps': 10, 
     'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
     'dim': 64,
     'learning_rate': 0.0002, 
     'distortion_fn': l2_distortion, 
     'latent_dim': 100, 
     'd_range': np.linspace(10, 80, 15)},
    {'dataset': 'MNIST', 
     'batch_size': 64, 
     'img_size': (32, 32, 1),  
     'steps': 10,
     'dim': 64, 
     'learning_rate': 0.0002, 
     'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
     'distortion_fn': l2_distortion, 
     'latent_dim': 100, 
     'd_range': np.linspace(10, 80, 15)},
    {'dataset': 'Gaussian', 
     'batch_size': 64, 
     'img_size': (32, 32, 1),
     'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),  
     'steps': 10, 
     'dim': 64,
     'learning_rate': 0.0002, 
     'distortion_fn': l2_distortion, 
     'latent_dim': 100, 'd_range': np.linspace(5, 50, 10)},
]
