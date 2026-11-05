# Spiking diffusion models for financial time series generation

Code for my Master of Mathematics (Computational Mathematics) research paper at the University of Waterloo:

**[Spiking Denoising Diffusion Probabilistic Model for Financial Time Series Data Generation](https://uwaterloo.ca/computational-mathematics/sites/default/files/uploads/documents/research-paper_seantang.pdf)**, Sean Tang, 2025. Supervised by Prof. Justin Wan and Prof. James Thompson.

## What this is

Diffusion models (DDPM) can generate realistic synthetic asset price paths, which are useful for strategy testing, risk management and portfolio construction. They are also expensive to run.

This project swaps the standard neural network inside a DDPM for a **spiking neural network** (SNN), which sends binary spikes instead of floating point values and so replaces most multiply-accumulate operations with cheaper additions. It then compares the spiking model (SDDPM) against a standard DDPM of the same size (about 84M parameters each).

## Results

Both models were trained on 100,000 simulated paths (64 steps, half a year) under two price models, each at low, medium and high volatility:

- **Geometric Brownian motion (GBM)**
- **Heston stochastic volatility**

The spiking model matched the standard model on:

- the distribution of log returns, including Heston's fat tails
- drift and volatility estimates (mean within 0.005 of the training data)
- European call and put prices from Monte Carlo on the generated paths, compared with Black-Scholes and Heston closed-form prices

It did this at **34-38% lower estimated energy per network pass** (2.55-2.73 mJ vs 4.13 mJ on 45nm hardware assumptions, using 4 spiking time steps).

The paper also introduces a tuning step: after warm-up training, the model keeps training in 10-epoch rounds until the drift and volatility of 50,000 generated paths match the training data within a set tolerance.

## Repository layout

| File | What it does |
| --- | --- |
| `main.py` | Builds the dataset, trains or loads a model, generates paths, and saves plots and metrics |
| `dataset.py` | Simulated GBM and Heston path datasets, standardized for training |
| `diffusion.py` | DDPM training step and sampling, for both the spiking and non-spiking models |
| `models/timeseries_unet.py` | 1D U-Net used by the standard DDPM |
| `models/spiking_time_unet.py` | 1D spiking U-Net (LIF neurons, pre-spiking residual blocks) used by SDDPM |
| `metrics.py` | Log-return, drift and volatility comparisons, and Monte Carlo vs Black-Scholes option prices |
| `main_spiking.sh`, `main_nonspiking.sh` | SLURM job scripts used on the compute cluster |

## Setup

Requires Python 3.12 and an NVIDIA GPU (the code runs on `cuda:0`).

```bash
pip install -r requirements.txt
```

## Usage

Create an output folder first; the script does not create it.

```bash
mkdir -p results

# Train the spiking model on GBM paths, tuning drift and volatility as it trains
python main.py --train --tune --spiking=True \
  --dataset=GBM --mu=0.05 --sigma=0.1 \
  --epochs=400 --batch_size=128 --n_samples=500000 \
  --folderdir=results

# Train the standard (non-spiking) model
python main.py --train --tune --spiking=False \
  --dataset=GBM --mu=0.05 --sigma=0.1 \
  --epochs=400 --batch_size=128 --n_samples=500000 \
  --folderdir=results
```

You can also run the shell scripts to train a model or use a model to generate asset price paths. They are set up for a SLURM cluster, so update the paths in them first.

Create your own training dataset or use the GBM or Heston training dataset. An API solution is in progress.

For Heston paths, use `--dataset=Heston --theta=0.3` (long-run volatility; the code squares it to get the variance).

To generate paths from a saved model instead of training, drop `--train` and pass `--resume_model=path/to/model.pt`.

Main options (see `main.py` for all of them):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--spiking` | `False` | `True` for the spiking model (SDDPM), `False` for the standard DDPM |
| `--dataset` | `GBM` | `GBM` or `Heston` |
| `--mu` | 0.05 | Drift for GBM |
| `--sigma` | 0.1 | Volatility for GBM |
| `--theta` | 0.1 | Long-run volatility for Heston |
| `--epochs` | 100 | Training epochs |
| `--batch_size` | 8 | Batch size |
| `--tune` | off | Check drift and volatility every 10 epochs and stop once both are within `--tolerance` |
| `--tolerance` | 0.005 | Allowed gap for tuning |
| `--n_samples` | 10000 | Paths to generate after training |
| `--resume`, `--resume_model` | off | Load a saved model |

Outputs in `--folderdir`: the saved model (`.pt`), `paths.npz` with the original and generated paths, and plots comparing paths, return distributions and option prices.

## Acknowledgments

We would like to thank the authors of [syops-counter](https://github.com/iCGY96/syops-counter), [SDDPM](https://github.com/AndyCao1125/SDDPM), [pytorch-DDPM](https://github.com/w86763777/pytorch-ddpm), and [spikingjelly](https://github.com/fangwei123456/spikingjelly). This project builds upon their open-source work.
