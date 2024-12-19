import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
import logging
from src.utils.metrics import l2_distortion, l1_distortion
import scipy

# Set up concise logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,  # Set to INFO for less verbose output
    handlers=[
        logging.StreamHandler()  # Logs to console
    ]
)

class NERDTrainer:
    """
    Trainer for Neural Estimator of the Rate-Distortion Function (NERD).
    Implements Algorithm 1 from the paper.
    """
    def __init__(self, generator, dataloader, distortion_fn, config):
        """
        Args:
            generator (nn.Module): Generator network G_θ to optimize.
            dataloader (DataLoader): DataLoader providing input data samples.
            distortion_fn (callable): Distortion function (e.g., L1 or L2 loss).
            config (dict): Configuration dictionary.
        """
        self.generator = generator
        self.dataloader = dataloader
        self.distortion_fn = distortion_fn
        self.config = config
        self.device = config['device']

        logging.info(f"Initializing NERDTrainer with device: {self.device}")

        # Move generator to the appropriate device
        self.generator.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.generator.parameters(), lr=config['learning_rate'])

        # Initialize wandb for experiment tracking
        wandb.init(project="NERD-Rate-Distortion", config=config)

    def compute_distortion(self, x, y, beta):
        x_exp = x.unsqueeze(0)  # Shape: [1, batch_size, ...]
        y_exp = y.unsqueeze(0)  # Shape: [1, batch_size, ...]
        distortion_vals = torch.clamp(self.distortion_fn(x_exp, y_exp), min=1e-5, max=100)
        kappa = torch.exp(torch.clamp(-beta * distortion_vals, min=-100, max=100))
        kappa_sum = torch.sum(kappa, dim=1)
        distortion = torch.mean(
            torch.sum(kappa * distortion_vals, dim=1) / kappa_sum
        ).item()
        return distortion

    def solve_beta_star(self, x, y, beta_range=(1e-5, 1e3), tol=1e-4, max_iter=50):
        """
        Find beta* using a manual bracketed search to satisfy distortion constraints.
        """
        beta_low, beta_high = beta_range
        for iteration in range(max_iter):
            beta_mid = (beta_low + beta_high) / 2

            # Compute distortion for beta_low, beta_mid, and beta_high
            distortion_mid = self.compute_distortion(x, y, beta_mid)

            # Log progress only every few iterations
            if iteration % 5 == 0:
                logging.info(f"Iteration {iteration}: beta_mid={beta_mid}, distortion_mid={distortion_mid:.4f}")

            # Check if mid satisfies tolerance
            if abs(distortion_mid - self.config['d']) < tol:
                logging.info(f"Converged to beta*={beta_mid}")
                return beta_mid

            # Adjust bounds
            if distortion_mid < self.config['d']:
                beta_low = beta_mid
            else:
                beta_high = beta_mid

        logging.warning(f"Maximum iterations reached. Returning beta_mid={beta_mid}")
        return beta_mid

    def train(self):
        """
        Train the generator using Algorithm 1.
        """
        self.generator.train()

        for step in range(self.config['steps']):
            epoch_loss = 0
            progress_bar = tqdm(self.dataloader, desc=f"Step {step + 1}/{self.config['steps']}")

            for batch_idx, batch in enumerate(progress_bar):
                # Load data batch
                x, _ = batch
                x = x.to(self.device)

                # Sample latent variables z ~ PZ (Gaussian prior)
                batch_size = x.size(0)
                z = torch.randn(batch_size, self.config['latent_dim']).to(self.device)

                # Generate samples y = G_θ(z)
                y = self.generator(z)

                try:
                    # Solve for β*
                    beta_star = self.solve_beta_star(x, y)

                    # Compute κ_{i,j}(β, θ)
                    kappa = torch.exp(-beta_star * self.distortion_fn(x.unsqueeze(0), y.unsqueeze(0)))

                    # Compute loss
                    loss = -torch.mean(torch.log(torch.mean(kappa, dim=1)))
                except Exception as e:
                    logging.error(f"Error during training step {step}, batch {batch_idx}: {e}")
                    raise

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update progress
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})

            # Log metrics to wandb
            avg_loss = epoch_loss / len(self.dataloader)
            wandb.log({"step": step + 1, "loss": avg_loss})
            logging.info(f"Step {step + 1}/{self.config['steps']}, Avg Loss: {avg_loss:.4f}")

    def save_model(self, path):
        """
        Save the trained generator model.
        """
        torch.save(self.generator.state_dict(), path)
        logging.info(f"Model saved at {path}")

    def load_model(self, path):
        """
        Load a trained generator model.
        """
        self.generator.load_state_dict(torch.load(path, map_location=self.device))
        logging.info(f"Model loaded from {path}")
