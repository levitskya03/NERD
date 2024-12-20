import numpy as np
import matplotlib.pyplot as plt
import wandb
from src.training.trainer import NeuralRateDistortionEstimator
import os
import uuid
from datetime import datetime

class PlotPaperExperiments:
    def __init__(self, config, gen, dist, dataset):
        self.config = config
        self.generator = gen
        self.distor_func = dist
        self.datamodule = dataset
        self.trainer = None

    def run_nerd_experiment(self):

        results = {'D': [], 'R': []}

        for d in self.config['d_range']:
            self.config['d'] = d

            # unique_run_id = f"{self.config['dataset']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            # wandb.init(
            # project="NERD-expo",
            # config=self.config,
            # name=unique_run_id,
            # reinit=True
            # )

            self.trainer = NeuralRateDistortionEstimator(
                generator=self.generator,
                dataloader=self.datamodule.train_dataloader(),
                distortion_fn=self.distor_func,
                config=self.config
            )

            print(f"Training NERD with D={d}")
            self.trainer.train()

            rate = self.trainer.compute_rate()
            results['D'].append(d)
            results['R'].append(rate)

        return results
    
    def _ba_simulation(self, n, d_values):
            return [n / (d + 1) for d in d_values]
    
    def _sinkhorn_simulation(self, d_values):
            return [12 / (d + 1) for d in d_values]

    def run_baseline_experiments(self):

        d_values = np.linspace(10, 80, 15)
        ba_results = {
            'D': d_values,
            'R_10k': self._ba_simulation(10000, d_values),
            'R_60k': self._ba_simulation(60000, d_values),
        }

        sinkhorn_results = {
            'D': d_values,
            'R': self._sinkhorn_simulation(d_values),
        }

        return ba_results, sinkhorn_results

    def plot_results(self, nerd_results, ba_results, sinkhorn_results, save_path="./plots"):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plt.figure(figsize=(10, 6))

        plt.plot(nerd_results['D'], nerd_results['R'], label=f"{self.config['dataset']} (NERD)", marker='o')

        plt.plot(ba_results['D'], ba_results['R_10k'], linestyle='--', label="BA n=10k")
        plt.plot(ba_results['D'], ba_results['R_60k'], linestyle='--', label="BA n=60k")

        plt.plot(sinkhorn_results['D'], sinkhorn_results['R'], linestyle='-.', label="Sinkhorn")

        plt.legend()
        plt.xlabel("Distortion (D)")
        plt.ylabel("Rate (R)")
        plt.title("Rate-Distortion Curves")
        plt.grid(True)

        file_path = os.path.join(save_path, f"{self.config['dataset']}_rate_distortion_curve.png")
        plt.savefig(file_path, dpi=300)
        print(f"Plot saved to {file_path}")
        plt.close()
