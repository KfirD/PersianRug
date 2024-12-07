# Persian Rug

This repository is for reproducing the plots (and corresponding data) in "The Persian Rug: solving toy models of superposition using large-scale symmetries".

## Setup

1. Create and activate a new conda environment:

```bash
conda create -n persian-rug python=3.10
conda activate persian-rug
```

2. Install the package and its dependencies in editable mode:
```bash
pip install -e .
```


## Training your own models
See the jupyter notebook getting_started.ipynb.

## Re-creating the plots in the paper
To re-train the models we trained, run:

```bash
python train_models.py
```

Then to make the plots, run:

```bash
python plotting_scripts/make_rps_plots.py
python plotting_scripts/make_hadamard_plots.py
python plotting_scripts/plot_matrix.py saved_models/recreate_trained_models/921
```

