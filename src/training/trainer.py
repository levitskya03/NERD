import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
import logging
import scipy
from src.utils.metrics import l2_distortion, l1_distortion

# Set up logging to both console and file
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,  # Reduced to INFO to reduce verbosity
    handlers=[
        logging.FileHandler("log.txt"),  # Logs detailed info to file
        logging.StreamHandler()  # Console displays only concise logs
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
            generator (nn.Module): Generator network G_\u03b8 to optimize.
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
        x_exp = x.unsqueeze(0)
        y_exp = y.unsqueeze(0)
        distortion_vals = torch.clamp(self.distortion_fn(x_exp, y_exp), min=1e-5, max=100)
        kappa = torch.exp(torch.clamp(-beta * distortion_vals, min=-100, max=100))
        kappa_sum = torch.sum(kappa, dim=1)

        if torch.any(kappa_sum == 0):
            logging.warning("Kappa sum is zero. Returning high distortion.")
            return float('inf')  # Return a high distortion value instead of NaN

        distortion = torch.mean(
            torch.sum(kappa * distortion_vals, dim=1) / kappa_sum
        ).item()

        if torch.isnan(torch.tensor(distortion)):
            logging.warning(f"Distortion computation returned NaN. Replacing with high distortion value.")
            return float('inf')

        return distortion

    def solve_beta_star(self, x, y, beta_range=(1e-5, 10.0), tol=1e-4, max_iter=50):
        beta_low, beta_high = beta_range
        for iteration in range(max_iter):
            beta_mid = (beta_low + beta_high) / 2
            distortion_mid = self.compute_distortion(x, y, beta_mid)

            if abs(distortion_mid - self.config['d']) < tol:
                logging.info(f"Beta* converged: {beta_mid:.4f}")
                return beta_mid

            if distortion_mid < self.config['d']:
                beta_low = beta_mid
            else:
                beta_high = beta_mid

        logging.warning(f"Beta* did not converge. Returning beta_mid={beta_mid:.4f}.")
        return beta_mid

    def train(self):
        """
        Train the generator using Algorithm 1.
        """
        self.generator.train()

        for step in range(self.config['steps']):
            epoch_loss = 0
            total_batches = len(self.dataloader)

            # Set up tqdm for the current step
            progress_bar = tqdm(
                total=total_batches,
                desc=f"Step {step + 1}/{self.config['steps']}",
                dynamic_ncols=True,
                leave=True,
                unit="batch"
            )

            for batch_idx, batch in enumerate(self.dataloader):
                x, _ = batch
                x = x.to(self.device)

                batch_size = x.size(0)
                z = torch.randn(batch_size, self.config['latent_dim']).to(self.device)
                y = self.generator(z)

                try:
                    # Solve for β*
                    beta_star = self.solve_beta_star(x, y)

                    if beta_star is None:
                        logging.warning(f"Step {step + 1}, Batch {batch_idx + 1}: Beta* did not converge.")
                        continue

                    # Compute κ_{i,j}(β, θ)
                    kappa = torch.exp(
                        torch.clamp(-beta_star * self.distortion_fn(x.unsqueeze(1), y.unsqueeze(0)), min=-100, max=100)
                    )

                    if torch.any(torch.isnan(kappa)):
                        logging.warning(f"Step {step + 1}, Batch {batch_idx + 1}: NaN detected in kappa. Resetting.")
                        kappa = torch.ones_like(kappa)

                    # Compute loss
                    loss = -torch.mean(torch.log(torch.mean(kappa, dim=1)))

                    if torch.isnan(loss):
                        logging.warning(f"Step {step + 1}, Batch {batch_idx + 1}: NaN detected in loss. Skipping.")
                        continue
                except Exception as e:
                    logging.error(f"Error at Step {step + 1}, Batch {batch_idx + 1}: {e}")
                    continue

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()

                # Update the progress bar
                progress_bar.update(1)

            # Close the progress bar
            progress_bar.close()

            # Log metrics to wandb
            avg_loss = epoch_loss / total_batches
            wandb.log({"step": step + 1, "loss": avg_loss})
            print(f"Step {step + 1}/{self.config['steps']}: Avg Loss = {avg_loss:.4f}")

    def compute_rate(self, generator, dataloader):
        """
        Compute the rate (mutual information estimate) based on the trained generator.
        """
        generator.eval()  # Set generator to evaluation mode
        total_rate = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                x, _ = batch
                x = x.to(self.config['device'])

                # Generate latent samples
                z = torch.randn(x.size(0), self.config['latent_dim']).to(self.config['device'])

                # Compute generator output
                y = generator(z)

                # Compute distortion between x and y
                distortion = self.config['distortion_fn'](x, y)

                # Compute rate contribution
                rate = torch.mean(distortion).item()
                total_rate += rate * x.size(0)
                total_samples += x.size(0)

        return total_rate / total_samples
    

    def save_model(self, path):
        torch.save(self.generator.state_dict(), path)
        logging.info(f"Model saved at {path}")

    def load_model(self, path):
        self.generator.load_state_dict(torch.load(path, map_location=self.device))
        logging.info(f"Model loaded from {path}")
