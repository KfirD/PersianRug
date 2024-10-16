# pylint: disable=trailing-whitespace
"""
This module provides utilities for training, saving, loading, and measuring models and Hadamard models.
It includes functions to train single or multiple models with various hyperparameters, save the trained models,
load them from disk, and convert them into pandas DataFrames for further analysis.
Functions:
    train_model: Trains a single model with specified architecture and training parameters.
    train_multiple_models: Trains multiple models with various hyperparameter combinations and saves them.
    train_hadamard_model: Trains a single Hadamard model with specified configuration parameters.
    train_multiple_hadamard_models: Trains multiple Hadamard models with varying parameters and saves the results.
    _load_trained_models_as_list: Loads trained models of a specified type from a directory.
    _models_to_dataframe: Converts an iterable of models into a pandas DataFrame and a dictionary of models.
    load_trained_models: Loads trained models for a given experiment and returns them as a DataFrame.
    _load_trained_model_measurements_as_list: Loads trained model measurements from a specified experiment directory.
    _load_hadamard_model_measurements_as_list: Loads Hadamard model measurements from saved models.
    load_df_from_file: Loads a DataFrame and a dictionary from a file.
    _model_measurements_to_dataframe: Converts an iterable of ModelMeasurement objects into a pandas DataFrame and a dictionary.
    _hadamard_model_measurements_to_dataframe: Converts an iterable of HadamardModel instances into a pandas DataFrame and a dictionary.
    load_trained_model_measurements: Loads trained model measurements for a given experiment.
    load_hadamard_model_measurements: Loads Hadamard model measurements for a given experiment.
    load_multiple_trained_experiments_measurements: Generates a dataframe for multiple experiments.
    load_multiple_hadamard_experiments_measurements: Loads and combines measurements from multiple Hadamard experiments.
Constants:
    _IGNORE_SUFFIXES: A tuple of file suffixes to ignore when loading models from a directory.
"""

import itertools
import pathlib
import uuid
from typing import Any, Dict, Iterable, List, Tuple, Union

import dill
import numpy as np
import pandas as pd
import torch as t
from tqdm import tqdm

import data
import hadamard_model as hm
import model as m
from PersianRug.utils import model_measurement as mm
from PersianRug.utils import opt_linear as ol

ModelType = Union[m.Model, hm.HadamardModel]


# create/save models
def train_model(
    n_dense: int,
    n_sparse: int,
    p: float,
    activation: m.ActivationFunctionName = "relu",
    final_layer_bias: bool = False,
    tie_dec_enc_weights: bool = False,
    device: str = "cpu",
    train_win: bool = True,
) -> m.Model:
    """
    Trains a model with the given architecture and training parameters.

    Parameters:
        n_dense (int): Number of dense units in the model.
        n_sparse (int): Number of sparse units in the model.
        p (float): Probability parameter for data configuration.
        activation (m.ActivationFunctionName, optional): Activation function for the final layer. Default is "relu".
        final_layer_bias (bool, optional): Whether to include a bias term in the final layer. Default is False.
        tie_dec_enc_weights (bool, optional): Whether to tie the decoder and encoder weights. Default is False.
        device (str, optional): Device to run the model on, e.g., "cpu" or "cuda". Default is "cpu".
        train_win (bool, optional): Whether to train the initial layer weights and biases. Default is True.
    Returns:
        m.Model: The trained model.
    """
 

    # training variables for both models
    dataconfig = data.Config(p, n_sparse, (0, 1))
    datafactory = data.DataFactory(dataconfig)

    cfg = m.Config(
        # architecture parameters
        n_sparse=n_sparse,
        n_dense=n_dense,
        final_layer_bias=final_layer_bias,
        tie_dec_enc_weights=tie_dec_enc_weights,
        # training parameters
        data_size=30_000,
        batch_size=1024,
        max_epochs=1500,
        lr=(3e-3) / np.sqrt(n_sparse),
        update_times=3000,
        convergence_tolerance=0,
        loss_window=200,
    )

    cfg.final_layer_act_func = activation

    model = m.Model(cfg)
    model = model.to(device)
    if not train_win:
        model.initial_layer.weight.requires_grad_(False)

        if cfg.init_layer_bias:
            model.initial_layer.bias.requires_grad_(False)

    model.optimize(datafactory, device=device, plot=False, logging=False)
    return model


def train_multiple_models(
    n_sparses: Iterable[int],
    ratios: Iterable[float],
    p_feats: Iterable[float],
    experiment_name: str,
    activations: Iterable[m.ActivationFunctionName] = ("relu",),
    final_layer_biases: Iterable[bool] = (False,),
    tie_dec_enc_weights_list: Iterable[bool] = (False,),
    device: str = "cpu",
    overwrite_experiments: bool = False,
    train_win: bool = True,
):
    """
    Trains multiple models with various hyperparameter combinations and saves them.

    Parameters:
        n_sparses (Iterable[int]): List of sparse dimensions.
        ratios (Iterable[float]): List of ratios to determine the number of neurons.
        p_feats (Iterable[float]): List of feature probabilities.
        experiment_name (str): Name of the experiment for saving models.
        activations (Iterable[m.ActivationFunctionName], optional): List of activation functions. Default is ("relu").
        final_layer_biases (Iterable[bool], optional): List of booleans indicating if the final layer has biases. Default is (False).
        tie_dec_enc_weights_list (Iterable[bool], optional): List of booleans indicating if decoder and encoder weights are tied. Default is (False).
        device (str, optional): Device to train the model on. Default is "cpu".
        overwrite_experiments (bool, optional): If True, overwrite existing models in the experiment directory. Default is False.
        train_win (bool, optional): If True, train with windowing. Default is True.
    Returns:
        None
    """

    path = pathlib.Path(f"saved_models/{experiment_name}/")
    path.mkdir(parents=True, exist_ok=True)

    total = (
        len(n_sparses)
        * len(ratios)
        * len(p_feats)
        * len(activations)
        * len(final_layer_biases)
        * len(tie_dec_enc_weights_list)
    )
    pbar = tqdm(
        itertools.product(
            n_sparses,
            ratios,
            p_feats,
            activations,
            final_layer_biases,
            tie_dec_enc_weights_list,
        ),
        total=total,
    )
    pbar.set_description(
        """ratio: N/A, p_feat: N/A, non_linearity: N/A, 
        final_layer_biases: N/A, tie_dec_enc_weights: N/A,  
        loss: N/A
        """
    )
    for model_idx, (
        n_sparse,
        ratio,
        p_feat,
        activation,
        final_layer_bias,
        tie_dec_enc_weights,
    ) in enumerate(pbar):
        if (path / str(model_idx)).exists() and not overwrite_experiments:
            continue
        model = train_model(
            max(int(n_sparse * ratio), 1),
            n_sparse,
            p_feat,
            activation=activation,
            final_layer_bias=final_layer_bias,
            tie_dec_enc_weights=tie_dec_enc_weights,
            device=device,
            train_win=train_win,
        )
        pbar.set_description(
            f"ratio: {model.ratio():.2f}, p: {p_feat:.4f}, non_linearity: {activation}, bias: {final_layer_bias}, tie_weights: {tie_dec_enc_weights}, loss: {model.final_loss():.4f}"
        )
        model.save(path / str(model_idx))


def train_hadamard_model(
    n_dense: int,
    n_sparse: int,
    p: float,
    device: str = "cpu",
    n_tries: int = 1,
    batch_size: int = 1024,
    max_epochs: int = 100,
    loss_window: int = 50,
    update_times: int = 3000,
    data_size: int = 10_000,
    convergence_tolerance: float = 0.0,
) -> hm.HadamardModel:
    """
    Trains a Hadamard model with the given configuration parameters.

    Parameters:
        n_dense (int): Number of dense features.
        n_sparse (int): Number of sparse features.
        p (float): Probability parameter for data configuration.
        device (str, optional): Device to run the model on, e.g., 'cpu' or 'cuda'. Default is 'cpu'.
        n_tries (int, optional): Number of times to train the model. Default is 1.
        batch_size (int, optional): Size of each batch for training. Default is 1024.
        max_epochs (int, optional): Maximum number of epochs for training. Default is 100.
        loss_window (int, optional): Window size for loss calculation. Default is 50.
        update_times (int, optional): Number of times to update the model. Default is 3000.
        data_size (int, optional): Size of the dataset. Default is 10,000.
        convergence_tolerance (float, optional): Tolerance for convergence. Default is 0.0.
    Returns:
        tuple: A tuple containing the best Hadamard model and a list of final losses for each model trained.
    """

    dataconfig = data.Config(p, n_sparse, (0, 1))
    datafactory = data.DataFactory(dataconfig)

    cfg = hm.HadamardConfig(
        n_sparse=n_sparse,
        n_dense=n_dense,
        data_size=data_size,
        batch_size=batch_size,
        max_epochs=max_epochs,
        lr=(3e-1) / np.sqrt(n_sparse),
        update_times=update_times,
        convergence_tolerance=convergence_tolerance,
        loss_window=loss_window,
    )

    models = []
    for _ in range(n_tries):
        model = hm.HadamardModel(cfg, device=device)
        model = model.to(device)
        model.optimize(datafactory, device=device, logging=False)
        models.append(model)

    losses = [model.losses[-1] for model in models]
    model = sorted(models, key=lambda x: x.losses[-1])[0]
    return model, losses


def train_multiple_hadamard_models(
    n_sparses: Iterable[int],
    ratios: Iterable[float],
    p_feats: Iterable[float],
    experiment_name: str,
    overwrite_experiments: bool = False,
    device: str = "cpu",
    n_tries: int = 1,
    batch_size: int = 1024,
    max_epochs: int = 100,
    loss_window: int = 50,
    update_times: int = 3000,
    data_size: int = 10_000,
):
    """
    Trains multiple Hadamard models with varying parameters and saves the results.

    Args:
        n_sparses (Iterable[int]): Iterable of sparse dimensions to use for training.
        ratios (Iterable[float]): Iterable of ratios to determine dense dimensions.
        p_feats (Iterable[float]): Iterable of feature probabilities.
        experiment_name (str): Name of the experiment for saving models and results.
        overwrite_experiments (bool, optional): If True, overwrite existing experiments. Defaults to False.
        device (str, optional): Device to use for training (e.g., "cpu" or "cuda"). Defaults to "cpu".
        n_tries (int, optional): Number of tries for training each model. Defaults to 1.
        batch_size (int, optional): Batch size for training. Defaults to 1024.
        max_epochs (int, optional): Maximum number of epochs for training. Defaults to 100.
        loss_window (int, optional): Window size for loss calculation. Defaults to 50.
        update_times (int, optional): Number of update times for training. Defaults to 3000.
        data_size (int, optional): Size of the dataset to use for training. Defaults to 10,000.

    Returns:
        None
    """

    path = pathlib.Path(f"saved_models/{experiment_name}/")
    # if not overwrite: assert not path.exists()
    path.mkdir(parents=True, exist_ok=True)

    losses = []
    pbar = tqdm(
        itertools.product(n_sparses, ratios, p_feats),
        total=len(n_sparses) * len(ratios) * len(p_feats),
    )
    pbar.set_description("n_sparse: N/A, Ratio: N/A, p_feat: N/A, Loss: N/A")
    for model_idx, (n_sparse, ratio, p_feat) in enumerate(pbar):
        if (path / str(model_idx)).exists() and not overwrite_experiments:
            continue
        n_dense = max(1, int(n_sparse * ratio))
        model, loss_list = train_hadamard_model(
            n_dense,
            n_sparse,
            p_feat,
            device=device,
            n_tries=n_tries,
            batch_size=batch_size,
            max_epochs=max_epochs,
            loss_window=loss_window,
            update_times=update_times,
            data_size=data_size,
        )
        losses.append(loss_list)
        pbar.set_description(
            f"n_sparse: {model.cfg.n_sparse}, Ratio: {model.ratio():.2f}, p_feat: {p_feat}, Loss: {model.final_loss():.4f}"
        )
        model.save(path / str(model_idx))
        with open(path / f"{model_idx}_losses.dill", "wb") as file:
            dill.dump(losses, file)
    return


# load models
_IGNORE_SUFFIXES = (".modelinfo", ".dill", ".dill_spec", ".DS_Store")


def _load_trained_models_as_list(
    experiment_name: str, model_type: ModelType, device: str = "cpu"
) -> List[Union[m.Model, hm.HadamardModel]]:
    """
    Load trained models of a specified type from a directory.

    Args:
        experiment_name (str): Name of the experiment (directory name).
        model_type (Type[Union[m.Model, hm.HadamardModel]]): The type of model to load.
        device (str, optional): The device to load the model onto. Defaults to "cpu".

    Returns:
        List[Union[m.Model, hm.HadamardModel]]: A list of loaded models.

    Raises:
        AssertionError: If the specified directory does not exist.
    """
    path = pathlib.Path(f"saved_models/{experiment_name}/")
    assert path.is_dir(), f"Directory not found: {path}"

    models = []
    for file in path.iterdir():
        if file.is_file() and not file.name.endswith(_IGNORE_SUFFIXES):
            new_model = model_type.load(file, map_location=t.device(device))
            models.append(new_model)

    return models


def _models_to_dataframe(
    models: Iterable[m.Model],
) -> Tuple[pd.DataFrame, Dict[uuid.UUID, m.Model]]:
    """
    Converts an iterable of models into a pandas DataFrame and a dictionary of models.
    The keys of the dictionary correspond to model_id's which can be found in the dataframe.
    (The idea is to (1) not store memory-intensive models in the dataframe and (2) have O(1)
    look up time whenever you need to access a model.)
    Args:
        models (Iterable[m.Model]): An iterable of model instances.
    Returns:
        Tuple[pd.DataFrame, Dict[uuid.UUID, m.Model]]: A tuple containing:
            A pandas DataFrame with the following columns:
                'model_id': A unique identifier for each model.
                'p_feat': The feature parameter of the model.
                'ratio': The ratio calculated by the model's ratio method.
                'n_dense': The number of dense layers from the model's configuration.
                'n_sparse': The number of sparse layers from the model's configuration.
                'final_loss': The final loss calculated by the model's final_loss method.
            A dictionary mapping unique UUIDs to the corresponding model instances.
    """

    model_dict = {uuid.uuid4(): model for model in models}
    df = pd.DataFrame(
        [
            {
                "model_id": model_id,  # Use a unique identifier
                "p_feat": model.p_feat,
                "ratio": model.ratio(),
                "n_dense": model.cfg.n_dense,
                "n_sparse": model.cfg.n_sparse,
                "final_loss": model.final_loss(),
            }
            for (model_id, model) in model_dict.items()
        ]
    )
    return df, model_dict


def load_trained_models(
    experiment_name: str, device="cpu"
) -> Tuple[pd.DataFrame, Dict[uuid.UUID, m.Model]]:
    """
    Load trained models for a given experiment and return them as a DataFrame.
    Assumes the files in saved_models/experiment_name correspond to trained models
    (as opposed to Hadamard models).
    Args:
        experiment_name (str): The name of the experiment whose models are to be loaded.
        device (str, optional): The device to load the models onto (e.g., "cpu" or "cuda").
                                Defaults to "cpu".
    Returns:
        Tuple[pd.DataFrame, Dict[uuid.UUID, m.Model]]: A tuple containing:
            A pandas DataFrame with the following columns:
                'model_id': A unique identifier for each model.
                'p_feat': The feature parameter of the model.
                'ratio': The ratio calculated by the model's ratio method.
                'n_dense': The number of dense layers from the model's configuration.
                'n_sparse': The number of sparse layers from the model's configuration.
                'final_loss': The final loss calculated by the model's final_loss method.
            A dictionary mapping unique UUIDs to the corresponding model instances.
    """

    models = _load_trained_models_as_list(experiment_name, m.Model, device=device)
    return _models_to_dataframe(models)


def _load_trained_model_measurements_as_list(
    experiment_name: str, device="cpu"
) -> List[mm.ModelMeasurement]:
    """
    Load trained model measurements from a specified experiment directory.
    Args:
        experiment_name (str): The name of the experiment whose models are to be loaded.
        device (str, optional): The device to map the models to (e.g., "cpu" or "cuda"). Defaults to "cpu".
    Returns:
        list: A list of ModelMeasurement objects for each valid model file in the experiment directory.
    Raises:
        AssertionError: If the specified experiment directory does not exist.
    """

    path = pathlib.Path(f"saved_models/{experiment_name}/")
    assert path.is_dir()
    model_measurements = []
    for file in path.iterdir():
        if file.is_file() and not file.name.endswith(_IGNORE_SUFFIXES):
            new_model = m.Model.load(file, map_location=t.device(device))
            model_measurements.append(mm.ModelMeasurement(new_model))
    return model_measurements


def _load_hadamard_model_measurements_as_list(
    experiment_name: str, device="cpu"
) -> List[mm.HadamardModelMeasurement]:
    """
    Load Hadamard model measurements from saved models.
    This function loads Hadamard model measurements from a specified directory
    corresponding to the given experiment name. It iterates through the files
    in the directory, loads each model, and appends the measurements to a list.
    Args:
        experiment_name (str): The name of the experiment whose models are to be loaded.
        device (str, optional): The device to map the models to (e.g., "cpu" or "cuda").
                                Defaults to "cpu".
    Returns:
        list: A list of HadamardModelMeasurement objects containing the measurements of the loaded models.
    Raises:
        AssertionError: If the specified path is not a directory.
    """

    path = pathlib.Path(f"saved_models/{experiment_name}/")
    assert path.is_dir()
    model_measurements = []
    for file in path.iterdir():
        if file.is_file() and not file.name.endswith(_IGNORE_SUFFIXES):
            new_model = hm.HadamardModel.load(file, map_location=t.device(device))
            model_measurements.append(mm.HadamardModelMeasurement(new_model))
    return model_measurements


def load_df_from_file(
    experiment_name: str,
) -> Tuple[pd.DataFrame, Dict[uuid.UUID, ModelType]]:
    """
    Load a DataFrame and a dictionary from a file. Works for both trained or Hadamard model.
    Args:
        experiment_name (str): The name of the experiment, which is used to
                               construct the file path.
    Returns:
        tuple: A tuple containing the DataFrame and the dictionary loaded from the file.
    Raises:
        AssertionError: If the file does not exist at the constructed path.
    """

    path = pathlib.Path(f"saved_models/{experiment_name}/df_and_dict.dill")
    if not path.is_file():
        raise FileNotFoundError(f"No file found at {path}")
    with path.open("rb") as f:
        df, mm_dict = dill.load(f)
    return df, mm_dict


def _model_measurements_to_dataframe(
    model_measurements: Iterable[mm.ModelMeasurement],
) -> Tuple[pd.DataFrame, Dict[uuid.UUID, mm.ModelMeasurement]]:
    """
    Converts an iterable of ModelMeasurement objects into a pandas DataFrame and a dictionary.
    Args:
        model_measurements (List[mm.ModelMeasurement]): A list of ModelMeasurement objects.
    Returns:
        Tuple[pd.DataFrame, Dict[uuid.UUID, mm.ModelMeasurement]]:
            - A pandas DataFrame where each row represents a model measurement with various attributes.
            - A dictionary mapping unique UUIDs to their corresponding ModelMeasurement objects.
    """

    mm_dict = {uuid.uuid4(): model_meas for model_meas in model_measurements}
    df = pd.DataFrame(
        [
            {
                "model_id": model_id,  # Use a unique identifier
                "p_feat": model_meas.p_feat,
                "ratio": model_meas.ratio(),
                "n_dense": model_meas.cfg.n_dense,
                "n_sparse": model_meas.cfg.n_sparse,
                "final_loss": model_meas.final_loss,
                "final_layer_bias": model_meas.cfg.final_layer_bias,
                "tie_dec_enc_weights": model_meas.cfg.tie_dec_enc_weights,
                "nonlinearity": model_meas.cfg.final_layer_act_func,
                "chi_pval": model_meas.chi_pval,
                "chi_varvar": model_meas.chi_varvar,
                "chi_meanvar": model_meas.chi_meanvar,
                "bias_var": model_meas.bias_var,
                "diag_mean": model_meas.diag_mean,
                "diag_var": model_meas.diag_var,
            }
            for (model_id, model_meas) in mm_dict.items()
        ]
    )
    return df, mm_dict


def _hadamard_model_measurements_to_dataframe(
    models: Iterable[hm.HadamardModel],
) -> Tuple[pd.DataFrame, Dict[uuid.UUID, mm.HadamardModelMeasurement]]:
    """
    Converts an iterable of HadamardModel instances into a pandas DataFrame and a dictionary.
    Args:
        models (Iterable[hm.HadamardModel]): An iterable of HadamardModel instances.
    Returns:
        Tuple[pd.DataFrame, Dict[uuid.UUID, mm.HadamardModelMeasurement]]:
            - A pandas DataFrame containing the model measurements.
            - A dictionary mapping unique UUIDs to HadamardModel instances.
    """

    model_dict = {uuid.uuid4(): model for model in models}
    df = pd.DataFrame(
        [
            {
                "model_id": model_id,  # Use a unique identifier
                "p_feat": model.p_feat,
                "ratio": model.ratio(),
                "n_dense": model.cfg.n_dense,
                "n_sparse": model.cfg.n_sparse,
                "final_loss": model.losses[-1],
            }
            for (model_id, model) in model_dict.items()
        ]
    )
    return df, model_dict


def load_trained_model_measurements(
    experiment_name: str, device="cpu", overwrite=False
) -> Tuple[pd.DataFrame, Dict[uuid.UUID, mm.ModelMeasurement]]:
    """
    Load trained model measurements for a given experiment.
    This function loads the measurements of a trained model from a specified experiment.
    If the `overwrite` flag is set to False (default), it will try to load a dataframe
    that was already computed for the experiment. Otherwise, it will generate the dataframe
    and save it (overwriting an old one if it has already been generated).
    Args:
        experiment_name (str): The name of the experiment whose model measurements are to be loaded.
        device (str, optional): The device to load the model on. Defaults to "cpu".
        overwrite (bool, optional): If True, reload the measurements and save them to a file. Defaults to False.
    Returns:
        tuple: A tuple containing:
            - df (DataFrame): A DataFrame containing the model measurements.
            - mm_dict (dict): A dictionary containing additional model measurement information.
    Raises:
        AssertionError: If the specified path does not exist.
    """

    path = pathlib.Path(f"saved_models/{experiment_name}/")
    assert path.is_dir()

    try:
        if overwrite:
            raise FileNotFoundError  # raise FileNotFoundError to force overwrite
        df, mm_dict = load_df_from_file(experiment_name)
    except FileNotFoundError:
        file = path / "df_and_dict.dill"
        mms = _load_trained_model_measurements_as_list(experiment_name, device=device)
        df, mm_dict = _model_measurements_to_dataframe(mms)
        with file.open("wb") as f:
            dill.dump((df, mm_dict), f)
    return df, mm_dict


def load_hadamard_model_measurements(
    experiment_name: str, device: str = "cpu", overwrite: bool = False
) -> Tuple[pd.DataFrame, Dict[uuid.UUID, mm.HadamardModelMeasurement]]:
    """
    Load Hadamard model measurements for a given experiment.
    If the `overwrite` flag is set to False (default), it will try to load a dataframe
    that was already computed for the experiment. Otherwise, it will generate the dataframe
    and save it (overwriting an old one if it has already been generated).
    Args:
        experiment_name (str): The name of the experiment.
        device (str, optional): The device to load the model on. Defaults to "cpu".
        overwrite (bool, optional): Whether to overwrite the existing measurements file. Defaults to False.
    Returns:
        tuple: A tuple containing:
            - df (pandas.DataFrame): The dataframe containing the measurements.
            - mm_dict (dict): A dictionary containing the measurements.
    """
    path = pathlib.Path(f"saved_models/{experiment_name}/")
    assert path.is_dir()

    try:
        if overwrite:
            raise FileNotFoundError
        df, mm_dict = load_df_from_file(experiment_name)
    except FileNotFoundError:
        file = path / "df_and_dict.dill"
        mm_list = _load_hadamard_model_measurements_as_list(
            experiment_name, device=device
        )
        df, mm_dict = _hadamard_model_measurements_to_dataframe(mm_list)
        with file.open("wb") as f:
            dill.dump((df, mm_dict), f)
    return df, mm_dict


def load_multiple_trained_experiments_measurements(
    experiments: Dict[str, Dict[str, Any]], overwrite: bool = False
) -> Tuple[pd.DataFrame, Dict[uuid.UUID, mm.ModelMeasurement]]:
    """
    Generate dataframe for multiple experiments.
    Args:
        experiments (Dict[str, Dict[str, Any]]): A dictionary where keys are experiment names and values are
            dictionaries containing experiment-specific parameters.
        overwrite (bool, optional): If True, existing dataframes will be overwritten. Defaults to False.
    Returns:
        Tuple[pd.DataFrame, Dict[uuid.UUID, mm.ModelMeasurement]]: A tuple containing:
            - A DataFrame with combined measurements from all experiments.
            - A dictionary mapping UUID's to model measurements.
    """

    combined_df = pd.DataFrame()
    combined_dict = {}

    for experiment_name, experiment_dict in experiments.items():
        df, mm_dict = load_trained_model_measurements(
            experiment_name, overwrite=overwrite
        )
        df = df.assign(**experiment_dict)
        df['experiment_name'] = experiment_name
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        combined_dict.update(mm_dict)

    return combined_df, combined_dict


def load_multiple_hadamard_experiments_measurements(
    dict_of_experiments: Dict[str, Dict[str, Any]], overwrite: bool = False
) -> Tuple[pd.DataFrame, Dict[uuid.UUID, mm.HadamardModelMeasurement]]:
    """
    Load and combine measurements from multiple Hadamard experiments.
    Args:
        dict_of_experiments (Dict[str, Dict[str, Any]]): A dictionary where keys are experiment names
            and values are dictionaries containing experiment-specific parameters.
        overwrite (bool, optional): If True, existing dataframes will be overwritten. Defaults to False.
    Returns:
        Tuple[pd.DataFrame, Dict[uuid.UUID, mm.HadamardModelMeasurement]]: A tuple containing:
            - A combined DataFrame of all experiment measurements.
            - A combined dictionary of all Hadamard model measurements, keyed by UUID.
    """

    combined_df = pd.DataFrame()
    combined_dict = {}

    for experiment_name, experiment_dict in dict_of_experiments.items():
        df, mm_dict = load_hadamard_model_measurements(
            experiment_name, overwrite=overwrite
        )
        df['experiment_name'] = experiment_name
        df = df.assign(**experiment_dict)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        combined_dict.update(mm_dict)
    return combined_df, combined_dict

def linear_model_from_row(row: pd.Series) -> ol.LinearModel:
    model = ol.LinearModel(int(row['n_dense']), int(row['n_sparse']), row['p_feat'])
    return model

def create_multiple_optimal_linear_models_df(
        n_sparses: Iterable[int],
        ratios: Iterable[float],
        p_feats: Iterable[float]
) -> pd.DataFrame:
    combinations = list(itertools.product(ratios, p_feats, n_sparses))
    lin_df = pd.DataFrame(combinations, columns=['ratio', 'p_feat', 'n_sparse'])
    lin_df['n_dense'] = lin_df[['ratio','n_sparse']].apply(
                        lambda x: max(1,int(x['n_sparse']*x['ratio'])), axis=1)

    lin_df['experiment_nickname'] = 'opt_linear'
    lin_df['experiment_name'] = 'optimal_linear'
    lin_df['final_loss'] = lin_df.apply(lambda x: linear_model_from_row(x).final_loss() , axis=1)
    
    return lin_df

# def load_hadamard_losses(experiment_name, device="cpu"):
#     path = pathlib.Path(f"saved_models/{experiment_name}/")
#     assert path.is_dir()
#     losses = []
#     for file in path.iterdir():
#         if file.is_file() and not file.name.endswith(
#             (".modelinfo", ".dill", ".dill_spec", ".DS_Store")
#         ):
#             new_model = hm.HadamardModel.load(file, map_location=t.device(device))
#             losses.append(new_model._final_loss)
#     return losses
