import argparse
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np

import utils.plot.plotdefaults
from model import Model
from utils.plot.plotfuncs import plot_matrix_vector
from hadamard_model import HadamardModel

def plot_matrix_histograms(A_full, B):
    """
    Plot histograms of the diagonal and off-diagonal elements of the matrix

    Args:
        A_full: Full matrix to analyze
        B: Bias vector
    """
    plt.figure()
    plt.title("Diagonal")
    plt.hist(A_full[np.eye(A_full.shape[0], dtype=bool)], bins=30)

    print(f'{np.mean(A_full[np.eye(A_full.shape[0], dtype=bool)]) = } {np.std(A_full[np.eye(A_full.shape[0], dtype=bool)]) = }')
    print(f'{np.mean(B) = } {np.std(B) = } {np.max(B) = } {np.min(B) = }')

    plt.figure()
    plt.title("Off Diagonal")
    plt.hist(A_full[np.logical_not(np.eye(A_full.shape[0], dtype=bool))], bins=100)

    print(f'{np.mean(A_full[np.logical_not(np.eye(A_full.shape[0], dtype=bool))]) = } {np.std(A_full[np.logical_not(np.eye(A_full.shape[0], dtype=bool))]) = }')
    print(f'{np.mean(A_full[np.logical_not(np.eye(A_full.shape[0], dtype=bool))]**2) * (A_full.shape[0] - 1) = }')
    print(f'{np.linalg.matrix_rank(A_full) = }')

def plot_model_matrix(model_input):
    """
    Plot matrix visualization for a given model or model path

    Args:
        model_input: str, Path object pointing to the model file, or Model/HadamardModel instance
    """
    if isinstance(model_input, (str, pathlib.Path)):
        path = pathlib.Path(model_input)
        # Try loading as HadamardModel first, fall back to Model if that fails
        try:
            model = HadamardModel.load(path)
        except:
            model = Model.load(path)

        # Load config from modelinfo file
        with path.with_suffix('.modelinfo').open('rb') as f:
            info = pickle.load(f)
            cfg = info['cfg']
            print(f'{info["p_feat"] = }')
            print(cfg)
    else:
        model = model_input
        cfg = model.cfg

    # Handle different model types
    if isinstance(model, HadamardModel):
        A_full = (model.Wout @ model.Win).detach().cpu().numpy()
        if not model.scalar_of_Identity:
            A_full = model.Adiag.reshape(-1, 1).detach().cpu().numpy() * A_full
        B = model.mean.detach().cpu().numpy() * np.ones(model.n_sparse)
    else:
        A_full = (model.final_layer.weight.data @ model.initial_layer.weight.data).detach().cpu().numpy()
        B = model.final_layer.bias.data.detach().cpu().numpy()[:30]

    A = A_full[:30,:30]
    B = B[:30]

    plot_matrix_vector(A, B, ratio=cfg.n_dense / cfg.n_sparse)
    plot_matrix_histograms(A_full, B)

def main():
    plt.rc('font', size=30)

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='The path to the file to plot')
    args = parser.parse_args()

    plot_model_matrix(args.path)
    plt.show()

if __name__ == '__main__':
    main()
