"""
This module provides various plotting functions for visualizing data in different formats.

Functions:
    plot_column_vs_r_and_p:
        Plot a heatmap over ratio and p for a specified column.

    plot_multiple_column_vs_r_and_p:
        Plot multiple heatmaps, each heatmap over r and p for every unique groupby value.

    plot_columns_simple:
        Plot one column over another from a dataframe.

    plot_matrix_vector:
        Plot heatmap of matrix and vector next to it. 
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt  # pylint: disable=no-member
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.gridspec import GridSpec

import utils.plot.plotdefaults

plt.rc("font", **{"size": 30})


def plot_column_vs_r_and_p(
    df: pd.DataFrame,
    column_name: str,
    title: Optional[str] = None,
    colorbar_label: str = "Loss Values",
    ax: Optional[plt.Axes] = None,
    norm=None,
):
    """Plot a heatmap over ratio and p for a specified column.

    Args:
        df (pd.DataFrame): Dataframe containing data
        column_name (str): Name of column to plot
        title (Optional[str], optional): Title of plot. Defaults to None.
        colorbar_label (str, optional): Label of for column. Defaults to "Loss Values".
        ax (Optional[plt.Axes], optional): An axis for subplots. Defaults to None.
        norm (_type_, optional): Normalize for normalizing over subplots. Defaults to None.
    """

    pivot = df.pivot(index="ratio", columns="p_feat", values=column_name)

    if ax is None:
        plt.figure(figsize=(6, 5))
        plt.pcolormesh(
            pivot.columns, pivot.index, pivot.values, cmap="viridis", norm=norm
        )
        plt.colorbar(label=colorbar_label)
        plt.ylabel("$n_d/n_s$")
        plt.xlabel("$p$")
        if title:
            plt.title(title)
        plt.show()
    else:
        ax.pcolormesh(
            pivot.columns, pivot.index, pivot.values, cmap="viridis", norm=norm
        )
        ax.set_ylabel("$n_d/n_s$")
        ax.set_xlabel("$p$")
        if title:
            ax.set_title(title)
    return


def plot_multiple_column_vs_r_and_p(
    df: pd.DataFrame,
    col_name: str,
    groupby_col: str,
    label: str,
    paper_directory: Optional[str] = None,
    show_plot: bool = False,
    orientation: Optional[tuple] = None,
    logscale: bool = False,
    filename_prefix: Optional[str] = None,
):
    """Plot multiple heatmaps, each heatmap over r and p. For every unique groupby value,
    you create a corresponding heatmap (for example, we often group by n_sparse).

    Args:
        df (pd.DataFrame): Dataframe containing data.
        col_name (str): Column to make the heatmap for.
        groupby_col (str): Column to 'group by'
        label (str): Label for plot
        paper_directory (Optional[str], optional): Directory to send plots to paper.
                                                    Defaults to None.
        show_plot (bool, optional): Display plot or not. Defaults to False.
        orientation (Optional[tuple], optional): Subplot orientation. Defaults to None.
        logscale (bool, optional): Plot on logscale or not. Defaults to False.
        filename_prefix (Optional[str], optional): Prefix for filenames. Otherwise the filename
                                            is determined by the column name. Defaults to None.

    Raises:
        ValueError: We expect the groupby values to be sortable in this function.
    """

    try:
        groupby_vals = sorted(df[groupby_col].unique())
    except ValueError as e:
        raise ValueError("Column not sortable.") from e

    if orientation:
        assert len(groupby_vals) == orientation[0] * orientation[1]
        fig, axes = plt.subplots(
            nrows=orientation[0],
            ncols=orientation[1],
            figsize=(6 * orientation[1], 5 * orientation[0]),
        )
        axes = axes.reshape(-1)
    else:
        fig, axes = plt.subplots(
            nrows=1, ncols=len(groupby_vals), figsize=(6 * len(groupby_vals), 5)
        )

    vmin = df[col_name].min()
    vmax = df[col_name].max()
    if logscale:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

    sm = ScalarMappable(cmap=plt.get_cmap("viridis"), norm=norm)
    sm.set_array([])

    for i, groupby_val in enumerate(groupby_vals):
        if groupby_col == "n_sparse":
            plot_column_vs_r_and_p(
                df[df[groupby_col] == groupby_val],
                col_name,
                title=f"$n_s= {groupby_val}$",
                colorbar_label=label,
                norm=norm,
                ax=axes[i],
            )
        else:
            plot_column_vs_r_and_p(
                df[df[groupby_col] == groupby_val],
                col_name,
                title=f"{groupby_val}",
                colorbar_label=label,
                norm=norm,
                ax=axes[i],
            )

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.86, 0.1, 0.01, 0.78])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(label)

    if filename_prefix:
        filename = f"{filename_prefix}_RPS_{col_name}.png"
    else:
        filename = f"RPS_{col_name}.png"

    filename = filename.replace(" ", "_")

    plt.savefig(f"../figures/{filename}", bbox_inches="tight", dpi=300)
    if paper_directory:
        plt.savefig(Path(paper_directory) / f"{filename}", bbox_inches="tight", dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()
    return


def plot_columns_simple(
    df,
    xcol: str,
    ycol: str,
    huecol: str,
    styles: dict,
    xlabel: str,
    ylabel: str,
    title: str,
    show: bool = False,
    filename: Optional[str] = None,
    paper_directory: Optional[str] = None,
):
    """This is for plotting one column over another from a dataframe (for example, for plotting
    loss as a function of ratio).

    Args:
        df (_type_): Dataframe with data.
        xcol (str): x column
        ycol (str): y column
        huecol (str): column for each line/color.
        styles (dict): styles for each line (keys/values for plt.plot)
        xlabel (str): xlabel
        ylabel (str): ylabel
        title (str): title
        show (bool, optional): Display plot or not. Defaults to False.
        filename (Optional[str], optional): Filename. Defaults to None.
        paper_directory (Optional[str], optional): Directory for sending plots to paper.
                                                     Defaults to None.
    """

    plt.figure(figsize=(14, 12))
    df_sorted = df.sort_values(by=xcol)

    # Define line styles and widths
    for hc, style in styles.items():
        if hc not in list(df["experiment_nickname"].unique()):
            continue
        data = df_sorted[df_sorted[huecol] == hc]
        if hc == "opt_linear":
            plt.plot(
                np.array(data[xcol]),
                np.array(data[ycol]),
                color=style["color"],
                label=style["label"],
            )
            continue
        plt.scatter(np.array(data[xcol]), np.array(data[ycol]), **style)

    plt.xlabel(xlabel, fontsize=50)
    plt.ylabel(ylabel, fontsize=50)
    plt.tick_params(labelsize=30)
    plt.title(title, fontsize=50)
    plt.legend(title="Model")
    # plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if filename:
        plt.savefig(f"../figures/{filename}.png", bbox_inches="tight", dpi=300)
        if paper_directory:
            plt.savefig(
                Path(paper_directory) / f"{filename}.png", bbox_inches="tight", dpi=300
            )
    plt.close()

    if show:
        plt.show()
    return


def plot_matrix_vector(A: np.ndarray, B: np.ndarray, ratio: float):
    """Plot heatmap of matrix and vector next to it (expects A.shape = (n,n) and  B.shape = (n,)).

    Args:
        A (np.ndarray): Matrix (e.g. W matrix)
        B (np.ndarray): Vector (e.g. bias)
        ratio: float
    """

# Combine A and B for unified color scaling
    width_unit = A.shape[0]
    assert A.shape[0] == A.shape[1]

    gap_size = int(width_unit * .05 + 1)
    tot_width = width_unit + gap_size * 2 + 1
    
    combined = np.hstack((A, B.reshape(-1, 1)))
    
    # Create a diverging colormap
    cmap = plt.cm.bwr  # Red-White-Blue colormap, reversed
    norm = TwoSlopeNorm(vmin=np.min(combined), vcenter=0, vmax=np.max(combined))
    
    # Determine the absolute maximum value for symmetric color scaling
    max_abs_val = max(abs(combined.min()), abs(combined.max()))
    print(combined.max(), combined.min())
    
    # Create the figure
    fig = plt.figure(figsize=(12, 12 * (tot_width / width_unit)))
    
    # Create GridSpec
    gs = GridSpec(width_unit, tot_width, figure=fig, wspace=0)
    
    # Create subplots
    ax1 = fig.add_subplot(gs[:, :width_unit])  # Matrix A
    ax2 = fig.add_subplot(gs[:, -1 - gap_size])  # Vector B
    cax = fig.add_subplot(gs[:, -1])  # Colorbar
    
    # Plot matrix A
    im1 = ax1.imshow(A, cmap=cmap, vmin=-max_abs_val, vmax=max_abs_val, aspect='auto')
    ax1.set_title('Matrix')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Plot vector B
    _ = ax2.imshow(B.reshape(-1, 1), cmap=cmap, vmin=-max_abs_val, vmax=max_abs_val, aspect='auto')
    ax2.set_title('Bias')
    ax2.set_yticks([])  # Remove y-axis ticks for B
    ax2.set_xticks([])  # Show only one x-tick for the vector
    
    # Remove spines between A and B
    # ax1.spines['right'].set_visible(False)
    # ax2.spines['left'].set_visible(False)
    
    # Add colorbar
    plt.colorbar(im1, cax=cax, orientation='vertical', label='Weight Value')
    
    # Adjust layout
    # plt.tight_layout()
    plt.suptitle(f"Model Structure $(n_d / n_s = {ratio:.2f})$")
    plt.show()
