"""
This script is for making heatmaps (over p_feat and ratio) for various quantities 
of interest for the trained model.
"""

from typing import Dict, Tuple, Optional


import pandas as pd
from PersianRug.utils import model_scan as ms
from PersianRug.utils.plot import plotfuncs
from PersianRug.utils.model_measurement import ModelMeasurement

EXPERIMENTS = {"non_average_run_09_26": {'experiment_label' : 'Trained Models (128,...,4096)'}, 
                   "non_average_run_8192_09_26": {'experiment_label': 'Trained Models (8192)'}}
PAPER_DIR = ('/Users/alexinf/Dropbox/apps/Overleaf/Sparsity'
                       ' and Computation in NNs/PersianRug/images/')
N_SPARSES_TO_KEEP = [128, 1024, 8192]

def make_rps_plots(dframe: pd.DataFrame,
                    meas_dict: Dict['UUID', ModelMeasurement],
                    paper_directory: Optional[str] = None,
                    orientation: Tuple[int, int] = (2, 3),
                    show_plot: bool = False):
    """
    Generate and save plots.
    Parameters:
        dframe : pd.DataFrame
            DataFrame containing the data to be plotted. Must include a 'model_id' column.
        meas_dict : Dict[UUID, ModelMeasurement]
            Dictionary mapping model UUIDs to their corresponding ModelMeasurement objects.
        paper_directory : Optional[str], default=None
            Directory where the plots will be saved. If None, plots will not be saved.
        orientation : Tuple[int, int], default=(2, 3)
            Tuple specifying the orientation of the plots (rows, columns).
        show_plot : bool, default=False
            If True, the plots will be displayed. If False, the plots will not be displayed.
    """
    
    
    cols_to_plot = {
        'chi_varvar': r'$\Delta\mathrm{var}(\nu)$',
        'chi_pval': r'$p_{\mathrm{KS}}$',
        'diag_var': r'$\Delta \mathrm{diag}(W)$',
        'bias_var': r'$\Delta b_i$',
        'chi_meanvar': r'$\Delta \mathrm{var}_X(\nu)$',
        'final_loss': r'$\mathrm{Loss}$'
        }
    for (col_name, col_label) in cols_to_plot.items():
        plotfuncs.plot_multiple_column_vs_r_and_p(dframe, col_name, groupby_col='n_sparse', label=col_label, paper_directory=paper_directory, orientation=orientation, show_plot=show_plot)
        plotfuncs.plot_multiple_column_vs_r_and_p(dframe, col_name, groupby_col='n_sparse', label=col_label, paper_directory=paper_directory, orientation=orientation, show_plot=show_plot, logscale=True)

    funcs_to_plot = {
        "Max Lyapunov": {
            "label": r"$\Lambda$",
            "func": lambda x: meas_dict[x].Lyapunov[1].item()
        }
    }

    for (f, f_dict) in funcs_to_plot.items():
        assert f not in dframe.keys()
        col_name = f
        col_label = f_dict["label"]
        dframe[col_name] = dframe['model_id'].apply(f_dict['func'])
        plotfuncs.plot_multiple_column_vs_r_and_p(dframe, col_name, groupby_col='n_sparse', label=col_label, paper_directory=paper_directory, orientation=orientation, show_plot=show_plot)
        plotfuncs.plot_multiple_column_vs_r_and_p(dframe, col_name, groupby_col='n_sparse', label=col_label, paper_directory=paper_directory, orientation=orientation, show_plot=show_plot, logscale=True)

    return

if __name__ == "__main__":
    df, mm_dict = ms.load_multiple_trained_experiments_measurements(EXPERIMENTS, overwrite=False)    
    df = df[df['n_sparse'].isin(N_SPARSES_TO_KEEP)]

    make_rps_plots(df, mm_dict, paper_directory=None, orientation=None)
    # make_paper_plots(df, mm_dict, paper_directory=PAPER_DIR, orientation=None)

