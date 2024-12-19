import argparse
import torch
from src.training.trainer import NERDTrainer
from src.utils.metrics import l2_distortion, l1_distortion
from src.experiments.paper_experiment import PAPER_EXPERIMENTS 

def parse_args():
    """
    Parse command-line arguments for configuring the experiment.
    """
    parser = argparse.ArgumentParser(description="Run NERD Rate-Distortion experiments")

    # Model and Training Arguments
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimensionality of the latent space Z')
    parser.add_argument('--dim', type=int, default=64, help='Base dimensionality for the Generator')
    parser.add_argument('--img_size', type=int, nargs='+', default=[28, 28, 1], help='Output image dimensions (H, W, C)')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate for the optimizer')
    parser.add_argument('--steps', type=int, default=10, help='Number of training steps (epochs)')
    parser.add_argument('--distortion_fn', type=str, default='l2', help='Distortion function (e.g., l2)')
    parser.add_argument('--dataset', type=str, choices=['MNIST', 'FashionMNIST', 'Gaussian'], default='MNIST', help='Dataset to use')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--beta', type=float, default=0.1, help='Initial beta value for the training algorithm')

    # Paper experiments flag
    parser.add_argument('--repeat', action='store_true', help='Repeat the experiments from the paper')

    return parser.parse_args()

def get_distortion_fn(name):
    """
    Get the distortion function by name.
    """
    if name == 'l2':
        return l2_distortion
    elif name == 'l1':
        return l1_distortion
    else:
        raise ValueError(f"Unsupported distortion function: {name}")

def run_experiment(config):
    """
    Run a single experiment based on the provided configuration.
    """
    print("Running experiment with configuration:", config)
    trainer = NERDTrainer(config)
    trainer.train()

def main():
    """
    Main function for running the experiment.
    """
    args = parse_args()

    if args.repeat:
        print("Repeating experiments from the paper...")
        for exp_config in PAPER_EXPERIMENTS:
            run_experiment(exp_config)
    else:
        # Parse image size as a tuple
        img_size = tuple(args.img_size)

        # Create user-provided configuration
        config = {
            'batch_size': args.batch_size,
            'latent_dim': args.latent_dim,
            'dim': args.dim,
            'img_size': img_size,
            'learning_rate': args.learning_rate,
            'steps': args.steps,
            'distortion_fn': get_distortion_fn(args.distortion_fn),
            'dataset': args.dataset,
            'device': torch.device(args.device),
            'beta': args.beta
        }

        run_experiment(config)

if __name__ == "__main__":
    main()
