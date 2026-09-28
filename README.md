# Joint Well-Log and Seismic Modeling

This repository provides code for predicting porosity, permeability, and water saturation from well logs and seismic data.

## Data Availability

The original dataset is confidential. A synthetic dataset is provided in `data/synthetic/` to demonstrate the workflow. It supports testing the code but does not reproduce the numerical results obtained using the confidential data.

## Installation

The code was tested with Python 3.9.23 and PyTorch 2.7.0 with CUDA 12.8. Dependency versions are listed in `requirements.txt`. In a Python 3.9 environment, install the dependencies:

```sh
python -m pip install -r requirements.txt
```

The code automatically uses a CUDA-enabled GPU when available; otherwise, it runs on the CPU. Memory requirements depend on dataset size.

## Usage

Run the training script directly:

```sh
python train.py
```

The script reads the included well-log data, validity masks, time-depth pairs, and seismic traces, then performs training and cross-validation. Model checkpoints, predictions, and evaluation metrics are saved to `outputs/run_1/`.

For a quick test:

```sh
python train.py --epochs 1 --n_splits 2
```

Use `python train.py --help` to view options for input paths, training settings, and output location. 
