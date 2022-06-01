import numpy as np
import torch.nn as nn
import ngt.utils.math.diffops as diffops
import ngt.utils.extract as extract
from typing import Callable, Literal, Tuple
from torch import Tensor
from ngt.tasks.types import ImplicitFunction


class SDF(ImplicitFunction):

    def __init__(self, functional: Callable, dim: int = 3):
        """_summary_

        Parameters
        ----------
        functional : Callable
            _description_
        dim : int, optional
            _description_, by default 3
        """        
        super(SDF, self).__init__(functional)
        self.dim = dim

    def normal(self, coords: Tensor, condition: Tensor = None) -> Tensor:      
        x = coords.requires_grad_()
        d = self(x, condition=condition)
        n = diffops.gradient(x, d, dim=self.dim)
        return n / n.norm(dim=-1, keepdim=True)

    def to_mesh(
        self, 
        condition: Tensor = None,
        res: int = 100, 
        max_coord: float = 1.0, 
        device: Literal['cpu', 'cuda'] = 'cpu'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        Parameters
        ----------
        res : int, optional
            _description_, by default 100
        max_coord : float, optional
            _description_, by default 1.0
        device : Literal['cpu', 'cuda'], optional
            _description_, by default 'cpu'

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            _description_
        """        
        return extract.extract_level_set(self.forward, self.dim, res, bound=max_coord, device=device, condition=condition)

