# pylint: disable=trailing-whitespace
"""
This module defines the HadamardConfig and HadamardModel classes for setting up and training 
a Hadamard model using PyTorch.

Classes:
    HadamardConfig: A configuration class for setting up and training a Hadamard model.
    HadamardModel: A PyTorch neural network module that uses Hadamard matrices for its 
    weight initialization.
"""

import pathlib
import random
from dataclasses import dataclass
from typing import List, Self, Union

import dill
import numpy as np
import torch as t
from jaxtyping import Float
from scipy.linalg import hadamard
from torch import Tensor

import data


@dataclass
class HadamardConfig:
    """
    HadamardConfig is a configuration class for setting up and training a Hadamard model.
    Attributes:
        n_sparse (int): Number of sparse features.
        n_dense (int): Number of dense features.
        n_hidden_layers (int): Number of hidden layers in the model. Default is 0.
        data_size (int): Size of the dataset. Default is 10,000.
        batch_size (int): Size of each batch for training. Default is 1,024.
        max_epochs (int): Maximum number of epochs for training. Default is 10.
        min_epochs (int): Minimum number of epochs for training. Default is 0.
        loss_window (int): Window size for calculating loss. Default is 1,000.
        lr (float): Learning rate for the optimizer. Default is 1e-5.
        convergence_tolerance (float): Tolerance for convergence. Default is 1e-1.
        update_times (int): Number of times to update the model. Default is 10.
        residual (bool): Whether to use residual connections. Default is False.
        importance (int): Importance factor for certain features. Default is 0.
        scalar_of_Identity (bool): When true, the parameter Adiag (corresponding
                                   to how much to scale W) is a scalar multiple of the identity.
                                   This corresponds to the case in the paper. When false,
                                   Adiag is allowed to be any diagonal matrix.
    """

    # architecture parameters
    n_sparse: int
    n_dense: int
    n_hidden_layers: int = 0

    # training parameters
    data_size: int = 10_000
    batch_size: int = 1024

    max_epochs: int = 10
    min_epochs: int = 0
    loss_window: int = 1000

    lr: float = 1e-5
    convergence_tolerance: float = 1e-1

    update_times: int = 10
    residual: bool = False
    importance: int = 0
    scalar_of_Identity: bool = True


class HadamardModel(t.nn.Module):
    """
    HadamardModel is a PyTorch neural network module that uses Hadamard matrices for its 
    weight initialization. It is designed to work with sparse and dense representations 
    and includes methods for optimization and loss computation.
    Attributes:
        cfg (HadamardConfig): Configuration object containing model parameters.
        n_sparse (int): Number of sparse features.
        n_dense (int): Number of dense features.
        Win (torch.Tensor): Input weight matrix.
        Wout (torch.Tensor): Output weight matrix.
        mean (torch.nn.Parameter): Mean parameter for conditioning epsilons.
        _final_loss (float): Final loss value after optimization.
        scalar_of_Identity (bool): Flag indicating if Adiag is a scalar.
        Adiag (torch.nn.Parameter): Diagonal matrix or scalar for scaling.
    Methods:
        __init__(self, cfg: HadamardConfig, device='cpu', **optim_kwargs): Initializes the 
            HadamardModel.
        set_mean(self, mean): Sets the mean parameter.
        forward(self, x): Forward pass of the model.
        compute_loss(self, x, labels): Computes the mean squared error loss.
        brute_force_optimize(self, x, labels, range_of_A: np.ndarray): Optimizes the model
            by brute force.
        set_A(self, A): Sets the Adiag parameter.
        ratio(self): Returns the ratio of dense to sparse features.
        final_loss(self): Returns the final loss value.
        W_matrix(self): Returns the product of Wout and Win matrices.
        optimize(self, data_factory: data.DataFactory, plot=False, logging=True, device='cpu'): 
            Optimizes the model using Adam optimizer.
        save(self, path: str): Saves the model to a file.
        load(cls, path: Union[str, pathlib.Path], map_location="cpu"): Loads the model from a file.
    """

    def __init__(self, cfg: HadamardConfig, device: str = "cpu"):
        super().__init__()

        self.cfg = cfg

        self.n_sparse = cfg.n_sparse
        self.n_dense = cfg.n_dense

        H = hadamard(cfg.n_sparse, dtype=np.float32)

        rand_rows = random.sample(range(cfg.n_sparse), cfg.n_dense)

        self.Win = H[rand_rows]
        self.Wout = self.Win.T / cfg.n_dense

        self.Win = t.tensor(self.Win).to(device)
        self.Wout = t.tensor(self.Wout).to(device)

        self.mean = t.nn.Parameter(t.tensor(0.0).to(device))
        self.Win = self.Win.to(device)
        self.Wout = self.Wout.to(device)
        self.mean = self.mean.to(device)

        self.losses = None
        self.p_feat = None
        self._final_loss = float("inf")

        self.scalar_of_Identity = cfg.scalar_of_Identity
        if self.scalar_of_Identity:
            self.Adiag = t.nn.Parameter(t.tensor(1.0))
        else:
            self.Adiag = t.nn.Parameter(t.ones(self.n_sparse, device=device))

    def forward(
        self, x: Float[Tensor, "batch n_sparse"]
    ) -> Float[Tensor, "batch n_sparse"]:
        """
        Perform the forward pass of the Hadamard model.
        Args:
            x (Float[Tensor, "batch n_sparse"]): Input tensor with shape (batch, n_sparse).
        Returns:
            Float[Tensor, "batch n_sparse"]: Output tensor after applying the Hadamard 
                transformation and ReLU activation function.
        """
        preactivs = (x @ self.Win.T) @ self.Wout.T
        preactivs = self.Adiag.reshape(-1, 1) * (preactivs + self.mean)
        return t.nn.functional.relu(preactivs)

    def compute_loss(
        self,
        x: Float[Tensor, "batch n_sparse"],
        labels: Float[Tensor, "batch n_sparse"],
    ) -> float:
        """
        Computes the mean squared error (MSE) loss between the predicted output and the true labels.
        Args:
            x (Float[Tensor, "batch n_sparse"]): Input tensor of shape (batch, n_sparse).
            labels (Float[Tensor, "batch n_sparse"]): True labels tensor of shape (batch, n_sparse).
        Returns:
            float: The computed MSE loss as a float value.
        """

        device = self.Win.device
        x = x.to(device)
        labels = labels.to(device)
        y = self(x)
        return t.nn.functional.mse_loss(y, labels).item()

    def ratio(self) -> float:
        """
        Calculate the ratio of dense to sparse elements.
        Returns:
            float: The ratio of the number of dense elements (`n_dense`) to the number of sparse elements (`n_sparse`).
        """
        return self.n_dense / self.n_sparse

    def final_loss(self) -> float:
        """
        Returns the final loss value.
        This method retrieves the final loss value that has been computed and stored
        during the training process.
        Returns:
            float: The final loss value.
        """

        return self._final_loss

    def W_matrix(self) -> Float[Tensor, "n_sparse n_sparse"]:
        """
        Computes the product of the output weight matrix (Wout) and the input weight matrix (Win).
        Returns:
            Float[Tensor, "n_sparse n_sparse"]: The resulting matrix from the multiplication 
                of Wout and Win.
        """

        return self.Wout @ self.Win

    def optimize(
        self, data_factory: data.DataFactory, logging: bool = True, device: str = "cpu"
    ) -> List[float]:
        """
        Optimize the model using the provided data factory.
        Args:
            data_factory (data.DataFactory): An instance of DataFactory to generate data loaders.
            logging (bool, optional): If True, logs the loss and other information during training. 
                Defaults to True.
            device (str, optional): The device to run the optimization on ('cpu' or 'cuda').
                Defaults to 'cpu'.
        Returns:
            List[float]: A list of loss values recorded during the optimization process.
        """

        optimizer = t.optim.Adam(self.parameters(), lr=self.cfg.lr, eps=1e-7)
        loss_function = t.nn.functional.mse_loss

        losses = []
        step_log = []

        step = 0
        epoch = 0
        tot_steps = self.cfg.max_epochs * self.cfg.data_size // self.cfg.batch_size + 1
        update_times = min(tot_steps, self.cfg.update_times)

        loss_change = float("inf")

        importance_weights = t.pow(
            t.arange(self.cfg.n_sparse) + 1, -self.cfg.importance / 2
        ).reshape((1, -1))

        while (
            loss_change > self.cfg.convergence_tolerance or epoch < self.cfg.min_epochs
        ) and epoch < self.cfg.max_epochs:

            if len(losses) > 2 * self.cfg.loss_window:
                loss_change = np.log10(
                    np.mean(losses[-2 * self.cfg.loss_window : -self.cfg.loss_window])
                ) - np.log10(np.mean(losses[-self.cfg.loss_window :]))

            # create data
            data_obj = data_factory.generate_data_loader(
                data_size=self.cfg.data_size,
                batch_size=self.cfg.batch_size,
                device=device,
            )
            iterator = iter(data_obj)
            for batch, labels in iterator:

                # compute outputs
                batch = batch.to(device)
                labels = labels.to(device)
                importance_weights = importance_weights.to(device)
                predictions = self(batch)

                # compute loss, on_loss, and loss_change
                loss = loss_function(
                    predictions * importance_weights, labels * importance_weights
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step += 1
                time_to_log = step % (tot_steps // update_times) == 0 or step == 1
                if time_to_log:
                    step_log.append(step)
                    losses.append(loss.item())

                if time_to_log and logging:
                    print(f"{loss_change=}")
                    print(f"{loss.item() = }")
                    print(f"{epoch=} {len(losses) = }")

            epoch += 1

        self.losses = losses
        self.p_feat = data_factory.cfg.p_feature
        self._final_loss = losses[-1]
        return losses

    def save(self, path: str):
        """
        Save the model's configuration, state dictionary, losses, and feature probabilities 
            to a specified path.
        Args:
            path (str): The file path where the model data will be saved. The main data will 
                        be saved with the given path, and additional model information will 
                        be saved with a '.modelinfo' suffix.
        The method performs the following steps:
        1. Creates a dictionary containing the model's configuration, state dictionary, losses,
           and feature probabilities.
        2. Saves this dictionary to the specified path using the `torch.save` function with
           `dill` as the pickle module.
        3. Creates a '.modelinfo' file at the same path and saves additional model information
           using `dill.dump`.
        """

        save_data = dict(
            cfg=self.cfg,
            state_dict=self.state_dict(),
            losses=self.losses,
            p_feat=self.p_feat,
        )
        t.save(save_data, path, pickle_module=dill)
        path = pathlib.Path(path)

        with path.with_suffix(".modelinfo").open("wb") as f:
            dill.dump(
                dict(
                    cfg=self.cfg, losses=self.losses, p_feat=self.p_feat, modelpth=path
                ),
                f,
            )

    @classmethod
    def load(cls, path: Union[str, pathlib.Path], map_location="cpu") -> Self:
        """
        Load a model from a specified path.
        Args:
            path (Union[str, pathlib.Path]): The path to the saved model file.
            map_location (str, optional): The device to map the model to. Defaults to "cpu".
        Returns:
            An instance of the model class with the loaded state.
        """

        model_data = t.load(path, map_location=map_location)
        model = cls(model_data["cfg"])
        model.load_state_dict(model_data["state_dict"])
        model.losses = model_data["losses"]
        model.p_feat = model_data["p_feat"]
        return model
 