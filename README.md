# Persian Rug

This repository is for reproducing the plots (and corresponding data) in "The Persian Rug: solving toy models of superposition using large-scale symmetries". 


## Re-creating the plots
To re-train the models we trained, run: 

```
python train_models.py
```

Then to make the plots, run:

```
python plotting_scripts/make_rps_plots.py
python plotting_scripts/make_hadamard_plots.py
python plotting_scripts/plot_matrix.py saved_models/recreate_trained_models/921
```

