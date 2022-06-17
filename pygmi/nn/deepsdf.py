import torch
import torch.nn as nn
import numpy as np
from typing import List
from torch import Tensor



class _MLP(nn.Module):

    def __init__(
        self,
        num_layers: int, 
        input_dim: int, 
        output_dim: int, 
        hidden_dim: int,
        skip_in: List[int],
        geometric_init: bool,
        activation: nn.Module
    ):
        super(_MLP, self).__init__()
        self.actvn = activation  
        hidden_sizes = [input_dim] + ([hidden_dim] * (num_layers - 1)) + [output_dim]
        self.num_layers = len(hidden_sizes)
        self.skip_conn = set(skip_in)

        self.linears = nn.ModuleList()
        for layer in range(1, self.num_layers):
            out_size = hidden_sizes[layer]
            if layer + 1 in self.skip_conn:
                out_size -= input_dim
            lin = nn.Linear(hidden_sizes[layer - 1], out_size)
            if geometric_init:
                if layer == self.num_layers - 1:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(hidden_sizes[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -1.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_size))
            self.linears.append(lin)

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for idx, layer in enumerate(self.linears[:-1]):
            if idx + 1 in self.skip_conn:
                h = torch.cat([h, x], dim=-1)
            h = self.actvn(layer(h))
        output = self.linears[-1](h)
        return output


class SmoothDeepSDFNet(_MLP):

    def __init__(
        self,
        input_dim: int = 3, 
        hidden_dim: int = 512,
        num_layers: int = 8, 
        skip_conn: List[int] = [4]
    ):
        """Softplus-activated MLP, as proposed in https://arxiv.org/abs/2002.10099.
        Default parameters and spherical weight initialization are as in original paper.

        Parameters
        ----------
        input_dim : int, optional
            _description_, by default 3
        hidden_dim : int, optional
            _description_, by default 512
        num_layers : int, optional
            _description_, by default 8
        skip_conn : List[int], optional
            _description_, by default [4]
        """        
        super(SmoothDeepSDFNet, self).__init__(
            num_layers, input_dim, 1, hidden_dim, skip_conn, True, nn.Softplus()
        )


class DeepReLUSDFNet(_MLP):
    
    def __init__(
        self,
        input_dim: int = 3, 
        hidden_dim: int = 512,
        num_layers: int = 8, 
        skip_conn: List[int] = [4]
    ):
        """ReLU-activated MLP, as proposed in https://arxiv.org/abs/1901.05103.
        Default parameters are as in original paper. Also features spherical
        weight initialization proposed in https://arxiv.org/abs/1911.10414.

        Parameters
        ----------
        input_dim : int, optional
            _description_, by default 3
        hidden_dim : int, optional
            _description_, by default 512
        num_layers : int, optional
            _description_, by default 8
        skip_conn : List[int], optional
            _description_, by default [4]
        """        
        super(SmoothDeepSDFNet, self).__init__(
            num_layers, input_dim, 1, hidden_dim, skip_conn, True, nn.ReLU()
        )