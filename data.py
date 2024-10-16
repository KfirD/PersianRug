import torch as t
from torch import Tensor
import torch
from torch.utils.data import DataLoader

import scipy.sparse as sparse
import numpy as np

from typing import Tuple, Callable, Union
from jaxtyping import Float
from dataclasses import dataclass

@dataclass
class Config:

    p_feature: Union[float, Tensor]
    
    n_input: int
    domain: Tuple[float,float]
    func: Callable = lambda x: x
    
    # if we want to draw from data where exactly n_on features are on
    # otherwise we draw from the binomial distribution
    n_on: int = None
    reverse_lightcone: bool = False

class DataFactory:
    
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        

    def generate_data(
        self, 
        batch_size: int,
        device='cpu'
        ) -> Tuple[Float[Tensor, "batch_size n_input"], Float[Tensor, "batch_size n_input"]]:
        
        # create random tensor of size (batch_size, n_input) from correct distribution
        a,b = self.cfg.domain

        unmasked_data = (b-a)*t.rand(batch_size, self.cfg.n_input, device=device) + a
        
        # create mask that will decide where things will be randomly zeroed out
        limiter = self.cfg.p_feature if isinstance(self.cfg.p_feature, float) else self.cfg.p_feature.view((1, self.cfg.n_input))

        if self.cfg.reverse_lightcone:
            mask = t.rand(batch_size, self.cfg.n_input, device=device) < limiter
            mask = self.cfg.func.reverse_lightcone(mask)
            data = t.where(mask, unmasked_data, 0)
        else:
            mask = t.rand(batch_size, self.cfg.n_input, device=device) > limiter
            data = t.where(mask, 0, unmasked_data)
        
        # create labels
        labels = self.cfg.func(data)
        
        return data, labels
        
        # apply mask and return

    def generate_first_order_data(
        self, 
        batch_size: int
        ) -> Tuple[Float[Tensor, "batch_size n_input"], Float[Tensor, "batch_size n_input"]]:
        a,b = self.cfg.domain
        data_list = []
        for _ in range((batch_size//self.cfg.n_input) + 1):
            data = (b-a)*t.rand(self.cfg.n_input) + a
            data = t.diag(data)
            data_list.append(data)
            
        data = t.cat(data_list)
        
        # create labels
        labels = self.cfg.func(data)
        
        return data, labels
        
        # apply mask and return
    
    def generate_data_loader(
        self, 
        data_size: int,
        batch_size: int,
        device='cpu',
        workers: int=0
        ) -> DataLoader:
        
        assert self.cfg.n_on == 1 or self.cfg.n_on is None

        if workers == 0:
            if self.cfg.n_on == None:
                data, labels = self.generate_data(data_size, device=device)
            elif self.cfg.n_on == 1: 
                data, labels = self.generate_first_order_data(data_size)
            else:
                raise ValueError("n_on > 1 not implemented yet")
            
            dataset = t.utils.data.TensorDataset(data, labels)
            data_loader = DataLoader(dataset, batch_size)
        
        else:
            dataset = FunctionDataIterator(data_size, self.generate_data)
            data_loader = DataLoader(dataset, batch_size, num_workers=5, pin_memory=True)
        
        return data_loader
        
        # apply mask and return


class FunctionDataIterator(torch.utils.data.IterableDataset):
    def __init__(self, data_size, data_func):
        super(FunctionDataIterator).__init__()
        self.data_size = data_size
        self.data_func = data_func
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            return iter(zip(*self.data_func(self.data_size)))
        else:  # in a worker process
            # split workload
            per_worker = self.data_size // worker_info.num_workers            
            # return iter(zip(*self.data_func(per_worker)))
            return self.generate(per_worker)

    def generate(self, amount):
        for _ in range(amount):
            yield self.data_func(1)


@dataclass
class FuncConfig:
    """
    Configures a circuit class. Currently only works for one layer
    """
    n_input: int 
    n_output: int

    window_size: int
    n_layer: int = 1

class SparseFunction:
    """
    Computes a n-layer Relu circuit sparse function.
    """
    def __init__(self, cfg: FuncConfig):
        self.cfg = cfg
        assert cfg.n_input == cfg.n_output # have to fix the n_input, n_output problems with multi-layers

        total_elements = cfg.n_output * cfg.window_size

        self.layers = []
        self.biases = [None] * self.cfg.n_layer

        for _ in range(cfg.n_layer):
            data = np.random.randn(total_elements).astype(np.float32)
            # create exactly window_size inputs per output
            i_indices = np.repeat(np.arange(self.cfg.n_output), self.cfg.window_size) 
            j_indices = np.random.randint(cfg.n_input, size=total_elements)

            # make sure that each output has at least one positive input
            # without modifying the distribution
            data = data.reshape((cfg.n_output, cfg.window_size))
            all_negative_mask = data.max(axis=1) < 0
            data[all_negative_mask, :] = -data[all_negative_mask, :]
            data = data.reshape(total_elements)

            # mat[i_indices[a], j_indices[a]] = data[a] for a = 0, ... total_elements - 1
            # collisions in indices are summed
            mat = sparse.coo_array((data, (i_indices, j_indices)), shape=(self.cfg.n_output, self.cfg.n_input))
            mat = mat.tocsr()

            self.layers.append(mat)

        front_to_back_lightcone = sparse.identity(self.cfg.n_output, dtype=np.float32)
        for mat in reversed(self.layers):
            front_to_back_lightcone = (mat.T != 0).astype(np.float32) @ front_to_back_lightcone
        
        self.front_to_back_lightcone = front_to_back_lightcone

    def __call__(self, x):
        x = x.detach().numpy().T

        for mat, bias in zip(self.layers, self.biases):
            x = mat @ x
            if bias is not None:
                x = x + bias
            x = x * (x > 0) #compute relu on the output
        
        x = x.T

        return t.from_numpy(x) 
    
    def sparsify_output(self, data, target_output_p):
        """
        Change the biases until the output probability is target_output_p. Do it with respect to the data.
        The batch size of data times target_output_p should be larger than 5 for enough statistics per output.
        """

        if isinstance(data, torch.Tensor):
            data = data.detach().numpy().T

        self.biases = []

        for layer in self.layers:
            target_activation_count = int(target_output_p * data.shape[1])
            assert target_activation_count > 5
            # pre_activation is (n_output, batch_size)
            pre_activation = layer @ data

            pre_activation.sort(axis=-1) # sort along the batch size output

            # get the activation halfway between the one which will give exactly 
            # target_output_p and the one which will give target_output_p + 1/batch_size
            bias = -(pre_activation[:, -target_activation_count] + pre_activation[:, -target_activation_count-1]) / 2
            self.biases.append(bias.copy().reshape(-1,1))

            pre_activation = layer @ data + self.biases[-1]
            data = pre_activation * (pre_activation > 0)
    
    def reverse_lightcone(self, mask):
        is_tensor = False
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().numpy()
            is_tensor = True

        out = ((self.front_to_back_lightcone @ mask.T) > 0).T
        if is_tensor:
            out = t.from_numpy(out)

        return out
        

@dataclass
class ParallelFuncConfig:
    n_input: int
    points_per_function: int

    functions_same: bool = True




class ParallelSparseFunction:
    def __init__(self, cfg: ParallelFuncConfig):
        self.cfg = cfg

        assert cfg.functions_same, "For now different functions aren't implemented."

        # self.x_locations = np.random.rand(self.cfg.points_per_function).astype(np.float32)
        # self.x_locations.sort()
        self.x_locations = np.linspace(0, 1, self.cfg.points_per_function + 1, endpoint=True).astype(np.float32)

        self.y_locations = np.random.rand(self.cfg.points_per_function + 1).astype(np.float32)
        self.y_locations[0] = 0

        self.left = 0
        self.right = np.random.rand()
    
    def __call__(self, x):
        is_tensor = False
        if isinstance(x, torch.Tensor):
            x = x.detach().numpy()
            is_tensor = True

        out = np.interp(x, self.x_locations, self.y_locations, left=self.left, right=self.right).astype(np.float32)
        if is_tensor:
            return t.from_numpy(out)
        
        return out



if __name__ == '__main__':
    def main1():
        import matplotlib.pyplot as plt
        func = SparseFunction(FuncConfig(500, 500, 4, n_layer=1))
        df = DataFactory(Config(.01, 500, (0, 1), func))
        dat = df.generate_data(int(200/.01))[0]
        func.sparsify_output(dat, .01)
        dat = df.generate_data(int(2000/.01))[0]
        out = func(dat)
        numacts = (out > 0).sum(axis=0) / out.shape[0]
        numacts = numacts.detach().numpy()
        plt.hist(numacts, bins=30)
        plt.show()

        numacts_bad = np.argmin(numacts)

        print(f'{func.biases[0][numacts_bad] = }  {func.layers[0][[numacts_bad], :].todense() =}')
    
    def main():
        np.random.seed(1)
        
        func = SparseFunction(FuncConfig(5, 5, 2, n_layer=2))
        df = DataFactory(Config(.1, 5, (0, 1), func, reverse_lightcone=True))
        print(sparse.find(func.layers[0]))
        print(sparse.find(func.layers[1]))
        print(df.generate_data(1))

    main()