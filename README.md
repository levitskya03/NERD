# NERD: Neural Estimation of the Rate-Distortion Function

## Overview

This repository hosts the implementation of **NERD** (Neural Estimator of the Rate-Distortion function), a novel approach for estimating the rate-distortion function for large-scale datasets using deep neural networks. This method bridges the gap between empirical deep neural network-based compression techniques and the theoretical limits of lossy compression.

NERD reformulates the rate-distortion objective into a scalable optimization problem, leveraging neural networks to accurately and efficiently estimate the rate-distortion function, even for high-dimensional datasets.


## Getting Started

### Installation

Clone the repository:

```bash
git clone https://github.com/levitskya03/NERD.git
cd NERD
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

To repeat one-time result from paper for every dataset:

```bash
python main.py --repeat
```

To repeat plots from paper:

```bash
python main.py --repeat_plots
```


For details on available parameters, use:

```bash
python main.py --help
```

## Reference

If you use this work in your research, please cite:

```
@ARTICLE{lei2022neuralrd,
  author={Lei, Eric and Hassani, Hamed and Bidokhti, Shirin Saeedi},
  journal={IEEE Journal on Selected Areas in Information Theory}, 
  title={Neural Estimation of the Rate-Distortion Function With Applications to Operational Source Coding}, 
  year={2022},
  volume={3},
  number={4},
  pages={674-686},
  doi={10.1109/JSAIT.2023.3273467}
}
```
