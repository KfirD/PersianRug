"""
This is a script for plotting curves of loss versus ratio and loss versus p, 
for various models. 
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from typing import Iterable
import numpy as np
import pandas as pd
# Custom Packages
from utils import model_analytics as ma
from utils import model_scan as ms
from utils.plot import plotfuncs

LINEAR_N_SPARSES = [8192]
TRAINED_N_SPARSES = [8192]
HADAMARD_N_SPARSES = [8192]

TARGET_P_FEATS = [0.001, 0.01, 0.05, 0.1, 0.4, 0.6]
TARGET_RATIOS = [0.01, 0.1, 0.3, 0.5, 0.8]


GRID_WIDTH = 20
RATIO_MIN = 0.002001
RATIO_MAX = 0.8

P_MIN = 0.002001
P_MAX = 0.8

TRAINED_EXPERIMENTS = {
    "recreate_trained_models": {"experiment_label": "Trained Models"},
}

HADAMARD_EXPERIMENTS = {
    "recreate_Hadamard_8192": {
        "experiment_label": r"$\mathrm{Hadamard} (n_s=8192)$",
        "experiment_nickname": "h-8192",
    },
}

STYLES = {
    "h-8192": {
        "label": r"$\mathrm{Persian\ Rug}\ (n_s=8192)$",
        "color": "black",
        #    'linestyle': '-',
        #    'linewidth': 12,
        #    'alpha': 0.5,
        "marker": "x",
        "s": 150,
        #    'markersize': 12
    },
    "t-128": {
        "label": r"$\mathrm{Trained\ Model}\ (n_s=128)$",
        "color": "red",
        #   'linestyle': '-',
        #   'linewidth': 4,
        "alpha": 0.5,
        #   'marker': 'x',
        "s": 100,
        #   'markersize': 12
    },
    "t-1024": {
        "label": r"$\mathrm{Trained\ Model}\ (n_s=1024)$",
        "color": "green",
        #   'linestyle': '-',
        #   'linewidth': 4,
        "alpha": 0.5,
        #   'marker': 'D',
        "s": 100,
        #   'markersize': 12
    },
    "t-8192": {
        "label": r"$\mathrm{Trained\ Model}\ (n_s=8192)$",
        "color": "orange",
        #   'linestyle': '-',
        #   'linewidth': 4,
        "alpha": 0.5,
        #   'marker': '^',
        "s": 100,
        #   'markersize': 12
    },
    "opt_linear": {
        "label": r"$\mathrm{Optimal\ Linear\ Model}$",
        "color": "blue",
        #   'linestyle': '-',
        #   'linewidth': 4,
        #   'alpha': 0.7,
        "marker": "x",
        "s": 100,
        #   'markersize': 12
    },
}


def plot_loss_with_ratio(
    comb_df: pd.DataFrame,
    target_ps: Iterable[float],
):
    """
    Plot loss with ratio for various target p_feats.

    Args:
        comb_df (pd.Dataframe): Combined dataframe (may include hadamard, trained, linear)
        target_ps (Iterable[float]): target p_feats        
    """
    for target_p_feat in target_ps:
        combined_target_p_feat = ma.get_closest_models_df(
            comb_df, ["n_sparse", "experiment_name"], p_feat=target_p_feat
        )
        plotfuncs.plot_columns_simple(
            combined_target_p_feat,
            xcol="ratio",
            ycol="final_loss",
            huecol="experiment_nickname",
            styles=STYLES,
            xlabel=r"$n_d/n_s$",
            ylabel="Loss",
            title=r"Loss at $p=" + f"{target_p_feat}$",
            filename=f"all_line_plot_target_p_feat_{target_p_feat}",
        )


def plot_loss_with_p(comb_df: pd.DataFrame, target_rs: Iterable[float]):
    """
    Plot loss with p for various ratios.

    Args:
        comb_df (pd.Dataframe): Combined dataframe (may include hadamard, trained, linear)
        target_rs (Iterable[float]): target ratios
    """
    for target_ratio in target_rs:
        combined_target_ratio = ma.get_closest_models_df(
            comb_df, groupby_cols=["n_sparse", "experiment_name"], ratio=target_ratio
        )
        plotfuncs.plot_columns_simple(
            combined_target_ratio,
            xcol="p_feat",
            ycol="final_loss",
            huecol="experiment_nickname",
            styles=STYLES,
            xlabel=r"$p$",
            ylabel="Loss",
            title=r"Loss at $r=" + f"{target_ratio}$",
            filename=f"all_line_plot_target_ratio_{target_ratio}",
        )


if __name__ == "__main__":

    df, mm_dict = ms.load_multiple_trained_experiments_measurements(
        TRAINED_EXPERIMENTS, overwrite=False
    )
    df["experiment_label"] = df["n_sparse"].apply(
        lambda x: "Trained Model " + r"$(n_s =" + f"{x}" + r")$"
    )
    df["experiment_nickname"] = df["n_sparse"].apply(lambda x: "t-" + f"{x}")
    h_df, h_dict = ms.load_multiple_hadamard_experiments_measurements(
        HADAMARD_EXPERIMENTS, overwrite=False
    )

    ratios = np.linspace(RATIO_MIN, RATIO_MAX, GRID_WIDTH)
    p_feats = np.linspace(P_MAX, P_MAX, GRID_WIDTH)
    lin_df = ms.create_multiple_optimal_linear_models_df(
        LINEAR_N_SPARSES, ratios, p_feats
    )

    h_df_star = h_df[h_df["n_sparse"].isin(HADAMARD_N_SPARSES)]

    # Plot over all trained_models first (don't filter on n_sparses yet)
    combined_df = pd.concat([h_df_star, df, lin_df], axis=0)
    plot_loss_with_ratio(combined_df, TARGET_P_FEATS)

    combined_df = pd.concat([h_df_star, df], axis=0)
    plot_loss_with_p(combined_df, TARGET_RATIOS)

    # Plot only n_sparses of interest for the trained models
    df = df[df["n_sparse"].isin(TRAINED_N_SPARSES)]

    combined_df = pd.concat([h_df_star, df, lin_df], axis=0)
    plot_loss_with_ratio(combined_df, TARGET_P_FEATS)

    combined_df = pd.concat([h_df_star, df], axis=0)
    plot_loss_with_p(combined_df, TARGET_RATIOS)
