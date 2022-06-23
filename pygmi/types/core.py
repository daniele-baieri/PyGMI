import torch.nn as nn
from typing import Any, Callable
from torch import Tensor
from pygmi.utils import cat_points_latent


class ImplicitFunction(nn.Module):

    def __init__(self, approximator: Callable):
        """Base class for objects representing implicit functions

        Parameters
        ----------
        approximator : Callable
            Callable objects computing the function (e.g. a neural net)
        """        
        super(ImplicitFunction, self).__init__()
        self.F = approximator

    def forward(self, coords: Tensor, condition: Tensor = None, *args, **kwargs) -> Any:
        """Computes the represented function on a set of points. args and kwargs are 
        forwarded to `self.F`, while a condition vector can be paired to each given point.

        Parameters
        ----------
        coords : Tensor
            A Tensor of point coordinates, shape `B_1 x ... x B_n x S x D`
        condition : Tensor, optional
            A condition vector to be paired to each sample of points, 
            shape `B_1 x ... x B_n x N`, by default None

        Returns
        -------
        Any
            Function computed over point set `coords`
        """        
        x = coords if condition is None else cat_points_latent(coords, condition) 
        return self.F(x, *args, **kwargs)