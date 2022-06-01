import torch.nn as nn
from typing import Any, Callable
from torch import Tensor
from ngt.utils import cat_points_latent


class ImplicitFunction(nn.Module):

    def __init__(self, functional: Callable):
        super(ImplicitFunction, self).__init__()
        self.F = functional

    def forward(self, coords: Tensor, condition: Tensor = None, *args, **kwargs) -> Any:
        x = coords if condition is None else cat_points_latent(coords, condition) 
        return self.F(x, *args, **kwargs)