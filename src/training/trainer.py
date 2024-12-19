import torch
import torch.optim as optim
from tqdm import tqdm
import wandb

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
            config (dict): Configuration dictionary with the following keys:
                - 'latent_dim': Dimensionality of the latent space Z.
                - 'learning_rate': Learning rate for the optimizer.
                - 'steps': Number of training steps (epochs).
                - 'beta': Initial value for the β parameter.
                - 'device': Device to use ('cuda' or 'cpu').
        """
        self.generator = generator
        self.dataloader = dataloader
        self.distortion_fn = distortion_fn
        self.config = config
        self.device = config['device']

        # Move generator to the appropriate device
        self.generator.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.generator.parameters(), lr=config['learning_rate'])

        # Initialize wandb for experiment tracking
        wandb.init(project="NERD-Rate-Distortion", config=config)

    def train(self):
        """
        Train the generator using Algorithm 1.
        """
        self.generator.train()

        for step in range(self.config['steps']):
            epoch_loss = 0
            progress_bar = tqdm(self.dataloader, desc=f"Step {step + 1}/{self.config['steps']}")
            
            for batch in progress_bar:
                # Load data batch
                x, _ = batch
                x = x.to(self.device)

                # Sample latent variables z ~ PZ (Gaussian prior)
                batch_size = x.size(0)
                z = torch.randn(batch_size, self.config['latent_dim']).to(self.device)

                # Generate samples y = G_θ(z)
                y = self.generator(z)

                # Compute κ_{i,j}(β, θ) and solve for β*
                kappa = torch.exp(-self.config['beta'] * self.distortion_fn(x.unsqueeze(1), y.unsqueeze(0)))

                # Compute loss
                loss = -torch.mean(torch.log(torch.mean(kappa, dim=1)))

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
            print(f"Step {step + 1}/{self.config['steps']}, Loss: {avg_loss:.4f}")

    def save_model(self, path):
        """
        Save the trained generator model.
        Args:
            path (str): Path to save the model checkpoint.
        """
        torch.save(self.generator.state_dict(), path)
        print(f"Model saved at {path}")

    def load_model(self, path):
        """
        Load a trained generator model.
        Args:
            path (str): Path to load the model checkpoint.
        """
        self.generator.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")
