import torch
import torch.nn as nn
import numpy as np
from typing import List
from torch import Tensor

class NeRF_MLP(nn.Module):

    def __init__(
        self,
        num_layers: int, 
        input_dim: int,
        input_view_dim: int,
        output_dim: int, 
        hidden_dim: int,
        skip_in: List[int],
    ):       
        super(NeRF_MLP, self).__init__()

        self.input_dim = input_dim
        self.input_view_dim = input_view_dim
        self.actvn = nn.ReLU()  

        hidden_sizes = [input_dim + input_view_dim] + ([hidden_dim] * (num_layers - 1)) + [output_dim]
        self.num_layers = len(hidden_sizes)
        self.skip_conn = set(skip_in)

        self.linears = nn.ModuleList()
        for layer in range(1, self.num_layers):
            out_size = hidden_sizes[layer]
            if layer + 1 in self.skip_conn:
                out_size -= input_dim
            lin = nn.Linear(hidden_sizes[layer - 1], out_size)
            self.linears.append(lin)

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for idx, layer in enumerate(self.linears[:-1]):
            if idx + 1 in self.skip_conn:
                h = torch.cat([h, x], dim=-1)
            h = self.actvn(layer(h))
        output = self.linears[-1](h)
        return output


class NeRF(NeRF_MLP):

    def __init__(
        self, 
        num_layers: int = 8, 
        input_dim: int = 3, 
        input_view_dim: int = 3,
        output_dim: int = 4,
        hidden_dim: int = 256,
        skip_conn: List[int] = [4],
    ):
        """ReLU-activated MLP, as proposed in https://arxiv.org/pdf/2003.08934.pdf.
        Default parameters and weight initialization are as in original paper.

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
        super(NeRF, self).__init__(
            input_dim, 1, hidden_dim, hidden_layers, 30.0, 30.0, use_first_layer_init, w0_in_layer_init
        )