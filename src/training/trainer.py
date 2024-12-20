import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
import logging
from src.utils.metrics import l2_distortion, l1_distortion
import os

logging.basicConfig(
    level=logging.INFO,  # Set base level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log1.txt"),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)


class NeuralRateDistortionEstimator:
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

        self.generator.to(self.device)

        self.optimizer = optim.Adam(self.generator.parameters(), lr=config['learning_rate'])

        wandb.init(project="NERD-expo", config=config)
    

    def compute_distortion(self, x, y, beta):
        """
        Compute distortion with shape compatibility checks.
        """

        x_exp = x.unsqueeze(0)  # [batch_size, 1, channels, height, width]
        y_exp = y.unsqueeze(0)  # [1, batch_size, channels, height, width]

        distortion_vals = torch.clamp(self.distortion_fn(x_exp, y_exp), min=1e-5, max=100)
        kappa = torch.exp(torch.clamp(-beta * distortion_vals, min=-100, max=100))
        kappa_sum = torch.sum(kappa, dim=1) + 1e-6  # Avoid division by zero

        distortion = torch.sum(kappa * distortion_vals, dim=1) / kappa_sum

        if torch.any(torch.isnan(distortion)):
            distortion = torch.tensor(float('inf')).to(self.device)  

        return torch.mean(distortion).item()

    def solve_beta_star(self, x, y, beta_range=(1e-5, 10.0), tol=1e-4, max_iter=50):
        """
        Alternative algorithm to solve for beta using self.compute_distortion with SGD.

        Args:
            x (torch.Tensor): Input data tensor.
            y (torch.Tensor): Reconstructed data tensor.
            beta_range (tuple): Range of beta values to explore.
            tol (float): Tolerance for convergence.
            max_iter (int): Maximum number of iterations.

        Returns:
        float: Optimized beta value.
        """
        beta = torch.tensor((beta_range[0] + beta_range[1]) / 2, requires_grad=True, device=self.device)
        optimizer = torch.optim.SGD([beta], lr=0.1)

        for iteration in range(max_iter):
            optimizer.zero_grad()

            distortion = self.compute_distortion(x, y, beta.item())
            distortion_tensor = torch.tensor(distortion, device=self.device, dtype=torch.float32, requires_grad=True)
            loss = (distortion_tensor - self.config['d']).pow(2)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                beta.clamp_(*beta_range)

            if loss.item() < tol:
                logging.info(f"Beta converged to {beta.item()} at iteration {iteration + 1}")
                return beta.item()

        logging.warning(f"Beta did not converge within {max_iter} iterations. Returning {beta.item()}")
        return beta.item()  

    def train(self):
        """
        Train the generator using Algorithm 1.
        """
        self.generator.train()

        checkpoint_dir = self.config.get('checkpoint_dir', './checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        for step in range(self.config['steps']):
            epoch_loss = 0
            total_batches = len(self.dataloader)

            progress_bar = tqdm(
                total=total_batches,
                desc=f"Step {step + 1}/{self.config['steps']}",
                dynamic_ncols=True,
                leave=True,
                unit="batch"
            )

            for batch_idx, batch in enumerate(self.dataloader):
                x, _ = batch

                if x.dim() == 2:  # Handle dimension reshaping
                    x = x.view(x.size(0), 1, 32, 32)
                    tqdm.write(f"Reshaped x to {x.shape} for compatibility")

                x = x.to(self.device)

                batch_size = x.size(0)
                z = torch.randn(batch_size, self.config['latent_dim']).to(self.device)
                y = self.generator(z)

                try:
                    beta_star = self.solve_beta_star(x, y)

                    if beta_star is None:
                        tqdm.write(f"Step {step + 1}, Batch {batch_idx + 1}: Beta* did not converge.")
                        continue

                    kappa = torch.exp(
                        torch.clamp(-beta_star * self.distortion_fn(x.unsqueeze(0), y.unsqueeze(0)), min=-100, max=100) #x-1
                    )

                    if torch.any(torch.isnan(kappa)):
                        tqdm.write(f"Step {step + 1}, Batch {batch_idx + 1}: NaN detected in kappa. Resetting.")
                        kappa = torch.ones_like(kappa)

                    loss = -torch.mean(torch.log(torch.mean(kappa, dim=1)))

                    if torch.isnan(loss):
                        tqdm.write(f"Step {step + 1}, Batch {batch_idx + 1}: NaN detected in loss. Skipping.")
                        continue
                except Exception as e:
                    tqdm.write(f"Error at Step {step + 1}, Batch {batch_idx + 1}: {e}")
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                progress_bar.update(1)

            progress_bar.close()

            avg_loss = epoch_loss / total_batches
            wandb.log({"step": step + 1, "loss": avg_loss})
            tqdm.write(f"Step {step + 1}/{self.config['steps']}: Avg Loss = {avg_loss:.4f}")

            if (step + 1) % self.config.get('checkpoint_interval', 5) == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"model_step_{step + 1}.pth")
                self.save_model(checkpoint_path)
                logging.info(f"Checkpoint saved at {checkpoint_path}")

        self.save_model(os.path.join(checkpoint_dir, "final_model.pth"))

    def compute_rate(self):
        
        """
        Compute the rate (mutual information estimate) based on the trained generator.
        """
        self.generator.eval()
        total_rate = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in self.dataloader:
                x, _ = batch
                x = x.to(self.config['device'])

                z = torch.randn(x.size(0), self.config['latent_dim']).to(self.config['device'])

                y = self.generator(z)

                x = x.unsqueeze(0)
                y = y.unsqueeze(0)

                distortion = self.config['distortion_fn'](x, y)

                rate = torch.mean(distortion).item()
                total_rate += rate * x.size(0)
                total_samples += x.size(0)

        return total_rate / total_samples

    def save_model(self, path="./models"):
        """
        Save the entire NeuralRateDistortionEstimator state, including generator and optimizer.

        Args:
            path (str): File path where the model will be saved.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            'generator_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }

        try:
            torch.save(state, path)
            logging.info(f"Model saved successfully at {path}")
        except Exception as e:
            logging.error(f"Failed to save model at {path}: {e}")



    def load_model(self, path):
        self.generator.load_state_dict(torch.load(path, map_location=self.device))
        logging.info(f"Model loaded from {path}")