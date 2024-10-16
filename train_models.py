"""
This script is for training the models we analyze in the paper.
"""

import pathlib
import shutil
from dataclasses import dataclass
from typing import List, Optional

import dill
import numpy as np
import torch as t
import utils.model_scan


@dataclass
class ExperimentSpec:
    """
    Data class for holding the experiment specification.
    """

    experiment_name: str
    n_sparses: np.ndarray
    grid_width: int
    ratios: np.ndarray
    p_feats: np.ndarray
    final_layer_biases: Optional[List]
    tie_dec_enc_weights: Optional[List]
    num_measurements: Optional[1]
    train_win: Optional[bool]
    num_samples: Optional[int]
    n_tries: Optional[int]
    batch_size: Optional[int]
    max_epochs: Optional[int]
    loss_window: Optional[int]
    update_times: Optional[int]
    data_size: Optional[int]

    def save(self):
        """
        Save the experiment specification using dill.

        Args:
            directory (str): The directory to save the file. Defaults to the current directory.

        Returns:
            str: The path to the saved file.
        """
        # Create a filename based on the experiment name
        path = pathlib.Path(f"saved_models/{self.experiment_name}/")
        # assert not path.exists()
        path.mkdir(parents=True, exist_ok=True)
        filename = f"{self.experiment_name}.dill_spec"
        filepath = path / filename

        # Save the entire object using dill
        with open(filepath, "wb") as f:
            dill.dump(self, f)

        return filepath

    @classmethod
    def load(cls, filepath: str):
        """
        Load an experiment specification from a dill file.

        Args:
            filepath (str): The path to the dill file.

        Returns:
            experiment_spec: The loaded experiment specification object.
        """
        with open(filepath, "rb") as f:
            return dill.load(f)


def main(overwrite: bool = True):
    """
    Train full models and then Hadamard model.
    """

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    grid_width = 20
    experiment_1 = "recreate_trained_models"
    path = pathlib.Path(f"saved_models/{experiment_1}/")
    if path.exists() and overwrite:
        shutil.rmtree(path)
    
    spec = ExperimentSpec(
        experiment_name=experiment_1,
        n_sparses=np.array([2**k for k in range(7, 12)]),
        grid_width=grid_width,
        ratios=np.linspace(0.002001, 0.8, grid_width),
        p_feats=np.linspace(0.002001, 0.8, grid_width),
        final_layer_biases=[True],
        tie_dec_enc_weights=[False],
        num_measurements=1,
        train_win=None,
        num_samples=None,
        n_tries=None,
        batch_size=None,
        max_epochs=None,
        loss_window=None,
        update_times=None,
        data_size=None,
    )

    spec.save()
    utils.model_scan.train_multiple_models(
        spec.n_sparses,
        spec.ratios,
        spec.p_feats,
        final_layer_biases=spec.final_layer_biases,
        tie_dec_enc_weights_list=spec.tie_dec_enc_weights,
        device=device,
        experiment_name=spec.experiment_name
    )

    grid_width = 20
    experiment_2 = "recreate_Hadamard_8192"

    path = pathlib.Path(f"saved_models/{experiment_2}/")
    if path.exists() and overwrite:
        shutil.rmtree(path)

    spec = ExperimentSpec(
        experiment_name=experiment_2,
        n_sparses=np.array([8192]),
        grid_width=grid_width,
        ratios=np.linspace(0.002001, 0.8, grid_width),
        p_feats=np.linspace(0.002001, 0.8, grid_width),
        final_layer_biases=None,
        tie_dec_enc_weights=None,
        num_measurements=None,
        train_win=False,
        num_samples=1,
        n_tries=5,
        batch_size=512,
        max_epochs=100,
        loss_window=100,
        update_times=3000,
        data_size=10_000,
    )
    spec.save()

    utils.model_scan.train_multiple_hadamard_models(
        spec.n_sparses,
        spec.ratios,
        spec.p_feats,
        spec.experiment_name,
        n_tries=spec.n_tries,
        batch_size=spec.batch_size,
        max_epochs=spec.max_epochs,
        loss_window=spec.loss_window,
        update_times=spec.update_times,
        data_size=spec.data_size,
        device=device,
    )
    
if __name__ == "__main__":
    OVERWRITE = True
    main(overwrite=OVERWRITE)
