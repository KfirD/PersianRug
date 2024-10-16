""" 
This module defines a neural network model with configurable architecture and training procedures.
Classes:
    Config: A dataclass that holds the configuration for a trained model, 
            including architecture and training parameters.
    Model: A neural network model class with methods for initialization, 
            forward pass, training, and evaluation.
"""
# pylint: disable=trailing-whitespace
import pathlib
from dataclasses import dataclass
from typing import List, Literal, Optional, Self, Union

import dill
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

import data

# For plotting live loss in jupyter 
try:
    from IPython.display import clear_output  # type: ignore
except ImportError:
    def clear_output(*args, **kwargs):
        """
        Dummy function for plotting live_losses when not in jupyter
        """
        pass

ActivationFunctionName = Literal["identity", "relu", "tanh", "leaky_relu", "gelu"]
act_funcs = {"identity": lambda x: x, "relu": F.relu, "tanh": F.tanh, 
             "leaky_relu": F.leaky_relu, "gelu":F.gelu}


@dataclass
class Config:
    """
    Configuration class for the model.
    Attributes:
        n_sparse (int): Number of sparse dimensions.
        n_dense (int): Number of dense dimensions.
        n_hidden_layers (int): Number of hidden layers. Default is 0.
        init_layer_bias (bool): Whether to initialize layer bias. Default is False.
        hidden_layers_bias (bool): Whether hidden layers have bias. Default is False.
        final_layer_bias (bool): Whether the final layer has bias. Default is False.
        init_layer_act_func (str): Activation function for the initial layer. Default is "identity".
        hidden_layers_act_func (str): Activation function for hidden layers. Default is "identity".
        final_layer_act_func (str): Activation function for the final layer. Default is "relu".
        tie_dec_enc_weights (bool): Whether to tie encoder and decoder weights. Default is False.
        data_size (int): Size of the dataset. Default is 10,000.
        batch_size (int): Size of each batch. Default is 1024.
        max_epochs (int): Maximum number of epochs for training. Default is 10.
        min_epochs (int): Minimum number of epochs for training. Default is 0.
        loss_window (int): Window size for loss calculation. Default is 1000.
        lr (float): Learning rate. Default is 1e-5.
        convergence_tolerance (float): Tolerance for convergence. Default is 1e-1.
        update_times (int): Number of times to update. Default is 10.
        residual (bool): Whether to use residual connections. Default is False.
        importance (int): Importance parameter. Default is 0.
    """

    # architecture parameters
    # dimensions
    n_sparse: int
    n_dense: int
    n_hidden_layers: int = 0

    # bias parameters
    init_layer_bias: bool = False
    hidden_layers_bias: bool = False
    final_layer_bias: bool = False
    
    # activations
    init_layer_act_func: str = "identity"
    hidden_layers_act_func: str = "identity"
    final_layer_act_func: str = "relu"

    # tie encoder/decoder
    tie_dec_enc_weights: bool = False
    
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

class Model(t.nn.Module):
    """
    Model class for a neural network with configurable architecture and training procedures.
    Attributes:
        cfg (Config): Configuration object containing model parameters.
        layers (nn.ModuleList): List of hidden layers in the model.
        p_feat (Optional): Placeholder for feature probabilities.
        losses (Optional): Placeholder for storing loss values during training.
        initial_layer (nn.Linear): Initial encoding layer transforming sparse input to dense representation.
        final_layer (nn.Linear): Final decoding layer transforming dense representation back to sparse output.
    Methods:
        __init__(cfg: Config):
            Initializes the model architecture based on the provided configuration.
        forward(x: Float[Tensor, "batch n_sparse"]) -> Float[Tensor, "batch n_sparse"]:
            Performs a forward pass through the network and returns the output tensor and a cache of activations.
        plot_live_loss(step_log, losses, sf_losses, steps):
            Plots live loss graphs during training.
        compute_loss(data_factory: data.DataFactory, batch_size=1000, device="cpu"):
            Computes the loss for a batch of data.
        optimize(data_factory: data.DataFactory, plot=False, logging=True, device='cpu') -> List[float]:
            Optimizes the model parameters using the Adam optimizer.
        ratio():
            Returns the ratio of dense to sparse dimensions.
        final_loss():
            Returns the final loss value after training.
        W_matrix():
            Returns the product of the final layer's weight matrix and the initial layer's weight matrix.
        save(path: str):
            Saves the model's state and configuration to the specified path.
        load(cls, path: Union[str, pathlib.Path], map_location="cpu"):
            Loads a model from the specified path.
        test_random_input(data_factory: data.DataFactory):
            Tests the model with random input data and prints the input, target, and prediction.
        test_first_feat():
            Tests the model with data where only the first feature is turned on and prints the input, label, and prediction.
        test_on_loss(data_factory: data.DataFactory, device, batch_size=1):
            Computes and returns the average loss for features that are turned on.
    """

    # ============================================================
    # Model Architecture
    # ============================================================
    
    def __init__(self, cfg: Config):
        super().__init__()

        self.cfg = cfg
        self.layers = nn.ModuleList()
        
        self.p_feat = None
        self.losses = None
                
        # encoding layer (n_sparse -> n_dense)
        self.initial_layer = nn.Linear(cfg.n_sparse, cfg. n_dense, bias = cfg.init_layer_bias)
        #self.layers.append(initial_layer)
        
        for _ in range(cfg.n_hidden_layers):
            self.layers.append(nn.Linear(cfg.n_dense, cfg.n_dense, bias = cfg.hidden_layers_bias))
            
        # decoding layer (n_dense -> n_sparse)
        self.final_layer = nn.Linear(cfg.n_dense, cfg.n_sparse, bias = cfg.final_layer_bias)
        
        if cfg.tie_dec_enc_weights:
            self.final_layer.weight.data = self.initial_layer.weight.data.T
        #self.layers.append(final_layer)
        
            
    def forward(self, x: Float[Tensor, "batch n_sparse"]) -> Float[Tensor, "batch n_sparse"]:
        """
        Perform a forward pass through the network.
        Args:
            x (Float[Tensor, "batch n_sparse"]): Input tensor with shape (batch, n_sparse).
        Returns:
            Tuple[Float[Tensor, "batch n_sparse"], dict]: Output tensor with shape (batch, n_sparse) and a cache dictionary containing activations at each layer.
        The forward pass includes:
        - Initial layer transformation followed by an activation function.
        - Sequential transformations through hidden layers with optional residual connections.
        - Final layer transformation followed by an activation function.
        - Caching of activations at each layer.
        """
        
        cache = {}                
        
        
        x = self.initial_layer(x)
        x = act_funcs[self.cfg.init_layer_act_func](x)
                
        cache["activations"] = [x]
        
        for l in self.layers:
            if self.cfg.residual:
                x = act_funcs[self.cfg.hidden_layers_act_func](l(x)) + x
            else:
                x = act_funcs[self.cfg.hidden_layers_act_func](l(x))
            cache["activations"].append(x)            
            
        x = self.final_layer(x)
        cache["activations"].append(x)
        x = act_funcs[self.cfg.final_layer_act_func](x)
        
        
        return x, cache
        
    # ============================================================
    # Model Training
    # ============================================================
    
    def plot_live_loss(self, step_log: List, losses: List, steps: int):
        """
        Plots the live loss and secondary loss during training.
        Parameters:
            step_log (list or array-like): A list or array of step numbers.
            losses (list or array-like): A list or array of loss values corresponding to the steps.
            on_losses (list or array-like): A list or array of secondary loss values corresponding to the steps.
            steps (int): The total number of steps for the x-axis limit.
        Returns:
            None
        """
        
        clear_output(wait=True)
        _, ax = plt.subplots(ncols=1, figsize=(10, 3))
                            
        ax[0].plot(step_log, losses)
        ax[0].set_title("loss")
        ax[0].set_xlabel("log steps")
        ax[0].set_ylabel("log loss")
        ax[0].set_xlim(0, steps)
        
        ax[0].text(0.5, -0.2, self.metadata, ha="center", va="top", transform=ax[0].transAxes, fontsize=12, color="gray")
        
        plt.show()
        return 

    def compute_loss(self, data_factory: data.DataFactory, batch_size: int = 1000, device: str = "cpu"):
        """
        Computes the loss for the model using the provided data factory.
        Args:
            data_factory (data.DataFactory): An instance of DataFactory to generate data loaders.
            batch_size (int, optional): The size of the batch to be used. Defaults to 1000.
            device (str, optional): The device to run the computations on ("cpu" or "cuda"). Defaults to "cpu".
        Returns:
            float: The mean loss computed over the batches.
        """


        data_obj = data_factory.generate_data_loader(
            data_size = batch_size, # single batch for now
            batch_size = batch_size,
            device = device
        )
        
        loss_function = F.mse_loss
        importance_weights = t.pow(t.arange(self.cfg.n_sparse) + 1, -self.cfg.importance / 2).reshape((1, -1))
        losses = []
        
        for batch, labels in data_obj:
            batch = batch.to(device)
            labels = labels.to(device)
            importance_weights = importance_weights.to(device)
            predictions, _ = self(batch)
            
            losses.append(loss_function(predictions * importance_weights, labels * importance_weights))            
        
        return np.mean(losses[0].item())

    
    def optimize(self, data_factory: data.DataFactory, plot: bool=False, logging: bool=True, device: str ='cpu') -> List[float]:
        """
        Optimize the model using the provided data factory.
        Args:
            data_factory (data.DataFactory): An instance of DataFactory to generate data loaders.
            plot (bool, optional): If True, plot the live loss during training. Defaults to False.
            logging (bool, optional): If True, print log information during training. Defaults to True.
            device (str, optional): The device to run the training on ('cpu' or 'cuda'). Defaults to 'cpu'.
        Returns:
            List[float]: A list of loss values recorded during the optimization process.
        """
            
        optimizer = t.optim.Adam([param for param in self.parameters() if param.requires_grad],
                                  lr = self.cfg.lr, eps=1e-7)
        loss_function = F.mse_loss
        
        losses = []
        on_losses = []
        step_log = []
        
        step = 0
        epoch = 0
        tot_steps = self.cfg.max_epochs*self.cfg.data_size//self.cfg.batch_size + 1
        update_times = min(tot_steps, self.cfg.update_times)

        loss_change = float("inf")
        
        importance_weights = t.pow(t.arange(self.cfg.n_sparse) + 1, -self.cfg.importance / 2).reshape((1, -1))
        
        while (loss_change > self.cfg.convergence_tolerance or epoch < self.cfg.min_epochs) and epoch < self.cfg.max_epochs:

            if len(losses) > 2 * self.cfg.loss_window:
                loss_change = np.log10(np.mean(losses[-2 * self.cfg.loss_window:-self.cfg.loss_window])) - np.log10(np.mean(losses[-self.cfg.loss_window:]))
            
            # create data
            data_obj = data_factory.generate_data_loader(
                data_size=self.cfg.data_size, 
                batch_size=self.cfg.batch_size,
                device=device
            )

            iterator = iter(data_obj)
            for batch, labels in iterator:
             
                batch = batch.to(device)
                labels = labels.to(device)
                importance_weights = importance_weights.to(device)
                predictions, _ = self(batch)
                
                loss = loss_function(predictions * importance_weights, labels * importance_weights)

                on_loss = t.tensor(1.0)
                
                # update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                step += 1
                # print if needed
                time_to_log = step % (tot_steps//update_times) == 0 or step == 1
                if time_to_log or not plot:
                    step_log.append(step)
                    losses.append(loss.item())
                    on_losses.append(on_loss.item())
                
                if time_to_log and logging:
                    print(f'{loss_change=}')
                    print(f'{loss.item() = }')
                    print(f'{epoch=} {len(losses) = }')
                    if plot:
                        self.plot_live_loss(step_log, losses, steps = len(step_log))
                                        
            epoch += 1
                    
        self.losses = losses
        self.p_feat = data_factory.cfg.p_feature
        return 
    
    # ============================================================
    # Helper functions
    # ============================================================
    
    def ratio(self)->float:
        """
        Calculate the ratio of dense to sparse configurations.
        Returns:
            float: The ratio of `n_dense` to `n_sparse` from the configuration.
        """

        return self.cfg.n_dense/self.cfg.n_sparse
    
    def final_loss(self)->Optional[float]:
        """
        Returns the final loss value from the list of losses.
        This method checks if there are any recorded losses. If there are, it returns
        the last loss value in the list. If there are no recorded losses, it returns None.
        Returns:
            float or None: The last loss value if available, otherwise None.
        """
        return self.losses[-1] if self.losses else None

    def W_matrix(self)->Float[Tensor, "n_sparse n_sparse"]:
        """
        Computes the product of the weight matrices of the final and initial layers.
        Returns:
            torch.Tensor: The resulting weight matrix after multiplying the weight 
            matrices of the final and initial layers.
        """
        return self.final_layer.weight.data @ self.initial_layer.weight.data
    
    
    def save(self, path: str):
        """
        Save the model's state and configuration to the specified path.
        Args:
            path (str): The file path where the model and its information will be saved.
        The method saves the model's state dictionary, configuration, losses, and feature parameters
        using the `torch.save` function with `dill` as the pickle module. Additionally, it creates a 
        separate file with a `.modelinfo` suffix containing the same information for easier access.
        The saved data includes:
            - cfg: The model's configuration.
            - state_dict: The state dictionary of the model.
            - losses: The losses associated with the model.
            - p_feat: The feature parameters of the model.
        """
        save_data = dict(cfg=self.cfg, state_dict = self.state_dict(), 
                         losses=self.losses, p_feat=self.p_feat)
        t.save(save_data, path, pickle_module=dill)
        path = pathlib.Path(path)

        with path.with_suffix('.modelinfo').open('wb') as f:
            dill.dump(dict(cfg=self.cfg,
                             losses=self.losses,
                             p_feat=self.p_feat,
                             modelpth=path), f)
    
    @classmethod
    def load(cls, path: Union[str, pathlib.Path], map_location="cpu") -> Self:
        """
        Load a model from a specified path.
        Args:
            path (Union[str, pathlib.Path]): The path to the saved model file.
            map_location (str, optional): The device to map the model to. Defaults to "cpu".
        Returns:
            Model: An instance of the model class with the loaded state.
        """
        model_data = t.load(path, map_location=map_location)
        model = cls(model_data["cfg"])
        model.load_state_dict(model_data["state_dict"])
        model.losses = model_data["losses"]
        model.p_feat = model_data["p_feat"]
        return model
    
