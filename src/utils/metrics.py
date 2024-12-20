import torch

def l2_distortion(x, y):
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x.shape={x.shape}, y.shape={y.shape}")
    return torch.mean((x - y) ** 2, dim=(1, 2, 3))


def l1_distortion(x, y):
    """
    Compute L1 distortion (mean absolute error) between x and y.
    Args:
        x (torch.Tensor): Ground truth tensor of shape (batch_size, ...).
        y (torch.Tensor): Generated tensor of shape (batch_size, ...).
    Returns:
        torch.Tensor: Mean absolute error for each sample in the batch.
    """
    return torch.mean(torch.abs(x - y), dim=(1, 2, 3))

