from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch as t
from scipy.stats import kstest

import data as scd
from model import Model

# Todo: (If n_sparse gets very large, and we consider only smaller n_dense):
#       Rewrite the functions to accept a model that keeps W_in, Wout unmultiplied.

def get_model_chi_data_all(model: Model, num_points: int = 1_000, batch_size: int = 1000, row: Optional[int]= None):
    """
    Generate chi data points for a given model.

    This function generates chi data points for a specified model by creating batches of data points
    and applying the model's weight matrix. The chi data points are computed as the product of the 
    generated data and the transposed weight matrix. If a specific row is provided, only that row of 
    the weight matrix is used.

    Parameters:
        model (Model): The model for which chi data points are generated.
        num_points (int): The total number of chi data points to generate.
        batch_size (int, optional): The number of data points to generate in each batch. Default is 1000.
        row (Optional[int], optional): The specific row of the weight matrix to use. If None, the entire 
                                       weight matrix is used. Default is None.

    Returns:
        data (torch.Tensor): The generated chi data points.
    """
    
    dataconfig = scd.Config(model.p_feat, model.cfg.n_sparse, (0, 1))
    datafactory = scd.DataFactory(dataconfig)
    
    # Get relevant model parameters
    W = model.W_matrix()
    if type(W) is t.Tensor:
        W = W.detach().clone().cpu()
        W.fill_diagonal_(0)
    else:
        W = t.tensor(W).float()
        W.fill_diagonal_(0)
    # get chi data points in batches    
    num_iters = int(np.ceil(num_points/batch_size))
    
    data = None
    for i in range(max(1,num_iters)):
        
        # generate chi data batch
        batch, _ = datafactory.generate_data(batch_size)
        if row is None:
            chi_batch = batch @ W.T
        else:
            chi_batch = batch @ W[row,:] # no transpose needed since a vector
        # append to data
        if data is None:
            data = chi_batch
        else:
            data = t.cat([data, chi_batch])
    return data


def get_model_gaussian_ks(model:Model, num_points: int = 1_000, batch_size: int = 1_000, row: Optional[int] = None):
    """
    Perform the Kolmogorov-Smirnov test for normality on chi data points.

    This function generates chi data points for a specified model and performs the Kolmogorov-Smirnov 
    test to check if the data follows a normal distribution. If a specific row is provided, the test 
    is performed on that row; otherwise, it is performed on all rows.

    Parameters:
        model (Model): The model for which chi data points are generated.
        num_points (int, optional): The total number of chi data points to generate. Default is 10,000.
        row (Optional[int], optional): The specific row of the weight matrix to use. If None, the test 
                                        is performed on all rows. Default is None.

    Returns:
        statistics (np.ndarray) or statistic (float): The KS statistics for each row or the specified row.
        p_values (np.ndarray) or p_value (float): The p-values for each row or the specified row.
    """
  
    if row is None:
        x = get_model_chi_data_all(model, num_points=num_points, batch_size=batch_size)   
    else:
        x = get_model_chi_data_all(model, num_points=num_points, batch_size=batch_size, row=row)

    x = x-x.mean(dim=0)
    x = x/x.std(dim=0)
    
    if row is None:
        n_sparse = x.shape[1]
        statistics = np.zeros(n_sparse)
        p_values = np.zeros(n_sparse)
        for i in range(n_sparse):
            statistics[i], p_values[i] = kstest(x[:, i], 'norm')
        return statistics, p_values

    statistic, p_value = kstest(x, 'norm')
    return statistic, p_value

# def chi_var(model: Model, row: Optional[int] = None, num_points = 1_000, batch_size = 1_000):
#     """
#     Compute the variance of chi data points for a given model.

#     This function generates chi data points for a specified model and computes their variance. If a 
#     specific row is provided, the variance is computed for that row; otherwise, it is computed for 
#     all rows.

#     Parameters:
#         model (Model): The model for which chi data points are generated.
#         row (Optional[int], optional): The specific row of the weight matrix to use. If None, the variance 
#             is computed for all rows. Default is None.

#     Returns:
#         variance (torch.Tensor): The variance of the chi data points.
#     """
#     chi_data = get_model_chi_data_all(model, num_points=num_points, batch_size=batch_size, row=row)
#     # chi_data.shape = (num_points, n_sparse) or (num_points, 1) if type(row)=int
#     return t.var(chi_data, axis=0)

def chi_var(model: Model, row: Optional[int] = None):
    x_var = model.p_feat/3 - (model.p_feat**2)/4
    if row:
        return (model.W_matrix()[row, :]**2).sum() * x_var 
    return (model.W_matrix()**2).sum(dim=1) * x_var

# def chi_mean(model: Model, row: Optional[int]= None, num_points = 1_000, batch_size = 1_000):
#     chi_data = get_model_chi_data_all(model, num_points=num_points, batch_size=batch_size, row=row) 
#     # chi_data.shape = (num_points, n_sparse) or (num_points, 1) if type(row)=int
#     return t.mean(chi_data, axis=0)

def chi_mean(model: Model, row: Optional[int] = None):
    x_mean = model.p_feat/2
    if row:
        return model.W_matrix()[row, :].mean() * x_mean
    return model.W_matrix().mean(dim=1) * x_mean

# def chi_varvar(model: Model, threshold: float = float('inf'), num_points = 1_000, batch_size = 1_000):
#     """
#     Compute the variance (over rows) of the variance (over data) of chi data points for a given model.

#     This function computes the variance of the variance of chi data points for a specified model. If 
#     the computed variance is below a given threshold, it is returned; otherwise, NaN is returned.

#     Parameters:
#     - model (Model): The model for which chi data points are generated.
#     - threshold (float, optional): The threshold for the variance of the variance. Default is infinity.

#     Returns:
#     - varvar (float): The variance of the variance of the chi data points, or NaN if it exceeds the threshold.
#     """
#     varvar = t.var(chi_var(model, num_points = num_points, batch_size=batch_size))
#     return varvar.item() if varvar < threshold else t.nan

def chi_varvar(model: Model):
    return t.var(chi_var(model)).item()

# def chi_mean_variance_over_rows(model: Model, num_points: int = 1_000, batch_size: int = 1_000):
#     return t.var(chi_mean(model, num_points = num_points, batch_size = batch_size)).item()

def chi_mean_variance_over_rows(model: Model):
    return t.var(chi_mean(model)).item()


def diag_var(model: Model, threshold: float = float('inf')):
    """
    Compute the variance (over rows) of the diagonal elements of the model's W matrix.

    This function computes the standard deviation of the diagonal elements of the model's W matrix.
    If the computed standard deviation is below a given threshold, it is returned; otherwise, NaN is returned.

    Parameters:
    - model (Model): The model for which the diagonal variance is computed.
    - threshold (float, optional): The threshold for the variance. Default is infinity.

    Returns:
    - ans (float): The standard deviation of the diagonal elements, or NaN if it exceeds the threshold.
    """

    W = t.tensor(model.W_matrix())
    W = W.detach()
    diags = W.diag()
    ans = diags.var().item()
    return ans if ans < threshold else t.nan


# Maybe use the same chi_data for all to save time eventually?
# def all_paper_measurements(model: Model, num_points: int = 1_000, batch_size: int = 1_000, row: int = 0):
#     chi_data = get_model_chi_data_all(model, num_points=num_points, batch_size=batch_size)
#     if row == None:
#         x = chi_data.clone()
#     else:
#         x = chi_data[:, row].clone()
        
#     x = x-x.mean()
#     x = x-x.var()
#     if row == None:
#         n_sparse = x.shape[1]
#         statistics = np.zeros(n_sparse)
#         p_values = np.zeros(n_sparse)
#         for i in range(n_sparse):
#             statistics[i], p_values[i] = kstest(x[:, i], 'norm')
#         statistics, p_values
#     else:
#         statistic, p_value = kstest(x, 'norm')
#         return statistic, p_value






def get_abseps3(eps_mat: t.Tensor):
    """ Returns the sum of the absolute cube of the elements of the input tensor (over dimension 1).
    Useful for calculating the Lyapunov exponent.

    Args:
        eps_mat (t.Tensor): Epsilon = W-diag(W)

    Returns:
        t.Tensor: The sum of the absolute cube of the elements of the input tensor (over dimension 1).
    """

    return t.sum(t.abs(eps_mat)**3, dim=1)

def get_eps2(eps_mat: t.Tensor):
    """ Returns the sum of the square of the elements of the input tensor (over dimension 1).
    Useful for calculating the Lyapunov exponent.

    Args:
        eps_mat (t.Tensor): Epsilon = W-diag(W)

    Returns:
        t.Tensor: The sum of the squares of the elements of the input tensor (over dimension 1).
    """
    return t.sum(eps_mat**2, dim=1)

def get_Lyapunov(eps_mat:t.Tensor):
    """Get Lyapunov coefficient for a given epsilon matrix.

    Args:
        eps_mat (t.Tensor): W-diag(W)

    Returns:
        t.tensor: Lyapunov coefficient
    """
    eps3 = get_abseps3(eps_mat) 
    eps2 = get_eps2(eps_mat)
    return eps3/eps2

def get_Lyapunov_min_max_avg(eps_mat):
    Lyapunov = get_Lyapunov(eps_mat)
    return t.min(Lyapunov), t.max(Lyapunov), t.mean(Lyapunov)

def get_eps_mat(W: t.Tensor):
    eps_mat = W * (1 - t.eye(W.size(0)))
    return eps_mat

def get_Lyapunov_min_max_avg_from_model(model):
    eps_mat = get_eps_mat(model)
    return get_Lyapunov_min_max_avg(eps_mat)

def estimate_rip_constant(A, k, num_samples=1000):
    _, n = A.shape
    delta_k = 0    
    for _ in range(num_samples):
        # Randomly select k columns
        cols = np.random.choice(n, k, replace=False)
        submatrix = A[:, cols]
        
        # Compute singular values
        s = np.linalg.svd(submatrix, compute_uv=False)
        
        # Update delta_k if necessary
        delta_k = max(delta_k, abs(1 - s.min()**2), abs(s.max()**2 - 1))    
    return delta_k

def mutual_coherence(model):
    W_in = model.initial_layer.weight.data.cpu()
    abs_dot_products = t.abs(W_in.T @ W_in)
    dinv = 1.0/t.diag(abs_dot_products)
    coherence_mat = dinv.reshape(-1,1) * abs_dot_products * dinv
    return t.max(coherence_mat-t.diag(t.diag(coherence_mat)))

def Delta_Eps(eps_mat):
    return t.var(t.mean(eps_mat, dim=1), dim=0)

def get_closest_models_df(df: pd.DataFrame, groupby_cols: Union[str, List[str]] = 'n_sparse', **targets: Dict[str, Union[float, int]]) -> pd.DataFrame:
    """
    For each "groupby_cols" (e.g. n_sparse), find the closest model where the metric is
    d(model_1, model_2) = sum_{t in targets} (model_1.t - model_2.t)^2. 

    Args:
        - df (pd.DataFrame): The input DataFrame containing model data.
        - groupby_cols (Union[str, List[str]]): Column(s) to group by. Defaults to 'n_sparse'.
        - **targets (Dict[str, Union[float, int]]): Target values for metrics to compare.

    Returns:
        pd.DataFrame: A DataFrame containing the closest models for each group.

    Raises:
        AssertionError: If any of the groupby_cols are not in the DataFrame,
                        or if no target values are provided,
                        or if any of the target columns are not in the DataFrame.

        ValueError: If a target column has an unsupported data type,
                    or if a numeric column receives a non-numeric target value.

    Example:
        To find the models with ratio closest to target_r and p_feat closest to target_p_feat
        for every n_sparse, run:
        get_closest_models_df(df, "n_sparse", ratio=target_r, p_feat=target_p_feat)
    """
    if isinstance(groupby_cols, str):
        groupby_cols = [groupby_cols]
    
    invalid_groupby_cols = set(groupby_cols) - set(df.columns)
    assert not invalid_groupby_cols, f"The following groupby columns are not in the DataFrame: {', '.join(invalid_groupby_cols)}"

    assert targets, "No target values provided. Please specify at least one target."
    assert targets.keys()
    invalid_target_columns = set(targets.keys()) - set(df.columns)
    assert not invalid_target_columns, f"The following target columns are not in the DataFrame: {', '.join(invalid_target_columns)}"

    distances=[]

    for col, target in targets.items():
        if np.issubdtype(df[col].dtype, np.number):
            if isinstance(target, (int, float)):
                distances.append((df[col] - target)**2)
            else:
                raise ValueError(f"Column '{col}' is numerical but received non-numeric target: {target}")
        else:
            raise ValueError(f"Unsupported data type for column '{col}': {df[col].dtype}")

    df['distance'] = sum(distances)
        
    min_distances = df.groupby(groupby_cols)['distance'].transform('min')
    min_distance_mask = df['distance'] == min_distances
    
    return df[min_distance_mask]

    



# 
# def get_model_chi_data_all_fast(epsop: EpsilonOperator, p_feat, n_sparse, num_points, batch_size = 1000, row = None):

#     dataconfig = scd.Config(p_feat, n_sparse, (0, 1))
#     datafactory = scd.DataFactory(dataconfig)
    
#     num_iters = int(np.ceil(num_points/batch_size))
    
#     data = None
#     for i in range(max(1,num_iters)):
        
#         # generate chi data batch
#         batch, _ = datafactory.generate_data(batch_size)
#         batch = batch
#         if row == None:
#             chi_batch = (batch @ epsop.T)
#         else:
#             chi_batch = batch @ (epsop[row,:]).T
#         # append to data
#         if data is None:
#             data = chi_batch
#         else:
#             data = t.cat([data, chi_batch])
#     return data



# (Potentially) Temporary Cuts:
# def get_abseps3_from_model(model):
#     W = model.W_matrix().detach().clone().cpu()
#     eps = W * (1 - t.eye(model.cfg.n_sparse).to())
#     ans = (t.sum(t.abs(eps)**3).detach())/(model.cfg.n_sparse)
#     return ans

# def get_eps2_from_model(model):
#     W = model.W_matrix().detach().clone().cpu()
#     eps = W * (1 - t.eye(W.size(0)))
#     ans = (t.sum(eps**2).detach())/(model.cfg.n_sparse)
#     return ans

# def get_eps_max_from_model(model):
#     W = model.W_matrix().detach().clone().cpu()
#     eps = W * (1 - t.eye(W.size(0)))
#     ans = (t.sum(eps**2).detach())/(model.cfg.n_sparse)
#     return ans

# def get_closest_model(models, ratio, p_feat):
#     data = [(i, model.ratio(), model.p_feat) for i,model in enumerate(models)]
#     distances = [(i, (r-ratio)**2+(p-p_feat)**2) for i,r,p in data]
#     distances.sort(key=lambda x: x[1])
#     return models[distances[0][0]]