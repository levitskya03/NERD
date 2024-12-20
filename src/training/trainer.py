import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
import logging
from src.utils.metrics import l2_distortion, l1_distortion

file_handler = logging.FileHandler("log.txt")
file_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.WARNING)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)

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

        self.generator.to(self.device)

        self.optimizer = optim.Adam(self.generator.parameters(), lr=config['learning_rate'])

        wandb.init(project="NERD-Rate-Distortion", config=config)
    

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

    def solve_beta_star(self, x, y, beta_range = (1e-5, 100.0), tol=1e-4, max_iter=50):
        logging.info(f"Starting solve_beta_star with x.shape={x.shape}, y.shape={y.shape}")

        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

        beta_low, beta_high = beta_range
        for iteration in range(max_iter):
            beta_mid = (beta_low + beta_high) / 2
            logging.info(f"Iteration {iteration + 1}: beta_mid={beta_mid}, beta_low={beta_low}, beta_high={beta_high}")

            try:
                distortion_mid = self.compute_distortion(x, y, beta_mid)
                logging.info(f"distortion_mid={distortion_mid}")
            except Exception as e:
                logging.error(f"Error computing distortion_mid: {e}")
                raise

            if abs(distortion_mid - self.config['d']) < tol:
                logging.info(f"Converged at iteration {iteration + 1} with beta_mid={beta_mid:.4f}, distortion_mid={distortion_mid}")
                return beta_mid

            if distortion_mid < self.config['d']:
                beta_low = beta_mid
            else:
                beta_high = beta_mid

        logging.warning(f"Beta* did not converge after {max_iter} iterations. Returning beta_mid={beta_mid:.4f}.")
        return beta_mid    



    def train(self):
        """
        Train the generator using Algorithm 1.
        """
        self.generator.train()

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

        self.save_model()

    def compute_rate(self, generator, dataloader):
        """
        Compute the rate (mutual information estimate) based on the trained generator.
        """
        generator.eval()
        total_rate = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                x, _ = batch
                x = x.to(self.config['device'])

                z = torch.randn(x.size(0), self.config['latent_dim']).to(self.config['device'])

                y = generator(z)

                x = x.unsqueeze(0)
                y = y.unsqueeze(0)

                distortion = self.config['distortion_fn'](x, y)

                rate = torch.mean(distortion).item()
                total_rate += rate * x.size(0)
                total_samples += x.size(0)

        return total_rate / total_samples

    def save_model(self, path="./models"):
        torch.save(self.generator.state_dict(), path)
        logging.info(f"Model saved at {path}")

    def load_model(self, path):
        self.generator.load_state_dict(torch.load(path, map_location=self.device))
        logging.info(f"Model loaded from {path}")
