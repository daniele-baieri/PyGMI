import numpy as np
import pygmi.utils.math.diffops as diffops
import pygmi.utils.extract as extract
from typing import Callable, Literal, Tuple
from torch import Tensor
from pygmi.types.core import ImplicitFunction


class SDF(ImplicitFunction):

    def __init__(self, approximator: Callable, dim: int = 3):
        """A utility class for objects representing signed distance fields.

        Parameters
        ----------
        approximator : Callable
            Function computing the SDF of given points
        dim : int, optional
            Dimensionality of domain, by default 3
        """        
        super(SDF, self).__init__(approximator)
        self.dim = dim

    def normal(self, coords: Tensor, condition: Tensor = None) -> Tensor:  
        """Computes SDF normals in query points `coords`, using the 
        normalized gradient of the SDF

        Parameters
        ----------
        coords : Tensor
            A Tensor of point coordinates, shape `B_1 x ... x B_n x S x D`
        condition : Tensor, optional
            A condition vector to be paired to each sample of points, 
            shape `B_1 x ... x B_n x N`, by default None

        Returns
        -------
        Tensor
            Shape `B_1 x ... x B_n x S x D`, normals of each point in `coords`
        """            
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
        """Converts `self` to a mesh using Marching Cubes.

        Parameters
        ----------
        condition : Tensor, optional
            Condition vector (for parametric SDFs), by default None
        res : int, optional
            Grid resolution, by default 100
        max_coord : float, optional
            Grid maximum absolute coordinate, by default 1.0
        device : Literal['cpu', 'cuda'], optional
            Device to run grid evaluation of SDF, by default 'cpu'

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Vertices and faces of output mesh
        """        
        return extract.extract_level_set(self.forward, self.dim, res, bound=max_coord, device=device, condition=condition)

