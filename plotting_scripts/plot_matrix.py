import argparse
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import utils.plot.plotdefaults
from model import Model
from utils.plot.plotfuncs import plot_matrix_vector

plt.rc('font', size=30)

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='The path to the file to plot')
args = parser.parse_args()

path = pathlib.Path(args.path)


model = Model.load(path)

A_full = (model.final_layer.weight.data @ model.initial_layer.weight.data).detach().cpu().numpy()
A = A_full[:30,:30]
B = model.final_layer.bias.data.detach().cpu().numpy()[:30]

with path.with_suffix('.modelinfo').open('rb') as f:
    info = pickle.load(f)
    cfg = info['cfg']

    print(f'{info["p_feat"] = }')
    print(cfg)


plot_matrix_vector(A, B, ratio=cfg.n_dense / cfg.n_sparse)

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
plt.show()
