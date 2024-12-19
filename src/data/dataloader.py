import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST
import numpy as np
from scipy.stats import ortho_group


class BaseDataModule:
    """
    Base class for all datamodule.
    """
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_dataloader(self, dataset, is_train=True):
        """
        Creates a DataLoader for a given dataset.
        Args:
            dataset: The dataset instance.
            is_train: Whether this loader is for training (enables shuffling).
        Returns:
            DataLoader instance.
        """
        return DataLoader(dataset, 
                          batch_size=self.batch_size, 
                          shuffle=is_train, 
                          pin_memory=self.device.type == 'cuda', 
                          num_workers=4)


class MNISTDataModule(BaseDataModule):
    """
    MNIST Data Module that handles loading and preprocessing of MNIST dataset.
    """
    def __init__(self, batch_size):
        super().__init__(batch_size)

    def train_dataloader(self):
        """
        Returns the DataLoader for training data.
        """
        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
        train_dataset = MNIST('./data', train=True, download=True, transform=transform)
        return self.create_dataloader(train_dataset, is_train=True)

    def test_dataloader(self):
        """
        Returns the DataLoader for test data.
        """
        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
        test_dataset = MNIST('./data', train=False, download=True, transform=transform)
        return self.create_dataloader(test_dataset, is_train=False)


class FashionMNISTDataModule(BaseDataModule):
    """
    FashionMNIST Data Module for loading and preprocessing the dataset.
    """
    def __init__(self, batch_size):
        super().__init__(batch_size)

    def train_dataloader(self):
        """
        Returns the DataLoader for training data.
        """
        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
        train_dataset = FashionMNIST('./data', train=True, download=True, transform=transform)
        return self.create_dataloader(train_dataset, is_train=True)

    def test_dataloader(self):
        """
        Returns the DataLoader for test data.
        """
        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
        test_dataset = FashionMNIST('./data', train=False, download=True, transform=transform)
        return self.create_dataloader(test_dataset, is_train=False)


class GaussianDataset(Dataset):
    """
    Custom dataset that generates samples from a multivariate Gaussian distribution.
    """
    def __init__(self, n_samples, m=1024, r=0.025, transform=None):
        """
        Args:
            n_samples: Number of samples to generate.
            m: Dimension of the Gaussian distribution.
            r: Decay rate for eigenvalues.
            transform: Optional transformation to apply.
        """
        self.transform = transform
        sigmas = 2 * np.exp(-r * np.arange(m))
        np.random.seed(seed=233423)
        U = ortho_group.rvs(m)  # Random orthogonal matrix
        cov_mat = U @ np.diag(sigmas**2) @ U.T  # Covariance matrix
        self.X = np.random.multivariate_normal(np.zeros(sigmas.shape[0]), cov_mat, n_samples)
        self.X = torch.tensor(self.X).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns a sample and a dummy label (e.g., 0) for compatibility.
        """
        return self.X[idx], torch.tensor(0)


class GaussianDataModule(BaseDataModule):
    """
    Data Module for Gaussian Dataset.
    """
    def __init__(self, batch_size, n_samples, m=1024, r=0.025):
        """
        Args:
            batch_size: Batch size for DataLoader.
            n_samples: Number of samples to generate for the dataset.
            m: Dimension of the Gaussian distribution.
            r: Decay rate for eigenvalues.
        """
        super().__init__(batch_size)
        self.dataset = GaussianDataset(n_samples=n_samples, m=m, r=r)

    def train_dataloader(self):
        """
        Returns DataLoader for training.
        """
        return self.create_dataloader(self.dataset, is_train=True)

    def test_dataloader(self):
        """
        Returns DataLoader for testing.
        """
        return self.create_dataloader(self.dataset, is_train=False)
