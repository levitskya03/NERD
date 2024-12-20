import argparse
import torch
from src.training.trainer import NERDTrainer
from src.utils.metrics import l2_distortion, l1_distortion
from src.experiments.paper_experiment import PAPER_EXPERIMENTS , PLOT_EXPERIMENTS
from src.experiments.paper_plots import PlotPaperExperiments
from src.models.generative_model import Generator  
from src.data.dataloader import MNISTDataModule, FashionMNISTDataModule, GaussianDataModule 
import wandb

def parse_args():
    """
    Parse command-line arguments for configuring the experiment.
    """
    parser = argparse.ArgumentParser(description="Run NERD Rate-Distortion experiments")

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
    parser.add_argument('--d', type=float, default=1.0, help='Distortion constraint for solving')

    parser.add_argument('--repeat', action='store_true', help='Repeat the experiments from the paper')
    parser.add_argument('--repeat_plots', action='store_true', help='Repeat the experiments with plots from the paper')

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
    
def return_instance(config):
    generator = Generator(img_size=config['img_size'], 
                          latent_dim=config['latent_dim'], 
                          dim=config['dim'])

    if config['dataset'] == 'MNIST':
        datamodule = MNISTDataModule(config['batch_size'])
    elif config['dataset'] == 'FashionMNIST':
        datamodule = FashionMNISTDataModule(config['batch_size'])
    elif config['dataset'] == 'Gaussian':
        datamodule = GaussianDataModule(config['batch_size'], 
                                         n_samples=60000,  # Default value
                                         m=config.get('gaussian_m', 1024),
                                         r=config.get('gaussian_r', 0.025))
    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")

    distortion_fn = config['distortion_fn']

    return generator, distortion_fn, datamodule

def run_experiment(config):
    """
    Run a single experiment based on the provided configuration.
    """
    print("Running experiment with configuration:", config)

    generator, distortion_fn, datamodule = return_instance(config)

    wandb.init(project="NERD-Rate-Distortion", config=config)

    trainer = NERDTrainer(generator=generator, 
                           dataloader=datamodule.train_dataloader(), 
                           distortion_fn=distortion_fn, 
                           config=config)
    trainer.train()
    wandb.finish()


def main():
    """
    Main function for running the experiment.
    """
    args = parse_args()

    if args.repeat:
        print("Repeating experiments from the paper...")
        for exp_config in PAPER_EXPERIMENTS:
            run_experiment(exp_config)
    elif args.repeat_plots:
        print("Repeating experiments with plots from the paper...")
        for exp_config in PLOT_EXPERIMENTS:
            generator, distortion_fn, datamodule = return_instance(exp_config)
            plotter = PlotPaperExperiments(config=exp_config, gen=generator, dist=distortion_fn, dataset=datamodule)
            nerd_results = plotter.run_nerd_experiment()
            ba_results, sinkhorn_results = plotter.run_baseline_experiments()
            plotter.plot_results(nerd_results, ba_results, sinkhorn_results)
    else:
        img_size = tuple(args.img_size)

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
            'beta': args.beta,
            'D': args.d
        }

        run_experiment(config)

if __name__ == "__main__":
    main()


 