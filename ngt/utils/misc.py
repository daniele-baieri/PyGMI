import torch
import functools
import time
import numpy as np
from torch import Tensor
from typing import Callable




def timer(func: Callable):
    """Decorator for timing functions.

    Parameters
    ----------
    func : Callable
        A function

    """    
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      
        run_time = end_time - start_time    
        print(f"Timed {func.__name__!r}: {run_time:.4f}s")
        return value
    return wrapper_timer

def sphere_sdf(x: Tensor, p: float = 2.0, r: float = 1.0) -> Tensor:
    """Signed distance function for a d-dimensional sphere.
    The sphere is assumed to be centered in the R^d origin.

    Parameters
    ----------
    x : Tensor
        Points: shape `B_1 x ... x B_n x d`
    p : float, optional
        Specify which L_p norm to compute, by default 2.0
    r : float, optional
        Radius of the sphere, by default 1.0

    Returns
    -------
    Tensor
        SDF values from the sphere for each point in `x`.
    """    
    return x.norm(dim=-1, p=p, keepdim=True) - r

def make_grid(resolution: int, bound: float, dim: int = 3) -> Tensor:
    """Instantiate a regular voxel grid with given bounds and resolution.

    Parameters
    ----------
    resolution : int
        Number of voxels in each dimension
    bound : float
        Maximum coordinate of voxel points
    dim : int, optional
        Number of dimensions, by default 3

    Returns
    -------
    Tensor
        Shape `N x dim`, each point representing a voxel corner in the 
        specified grid
    """    
    line = np.linspace(-bound, bound, resolution)
    grid = np.meshgrid(*([line] * dim))
    return torch.tensor(
        np.vstack([l.ravel() for l in grid]).T, dtype=torch.float
    )

def cat_points_latent(points: Tensor, latent: Tensor) -> Tensor:
    """Utility function to concatenate latent vectors of continuous
    data to finite samples thereof.

    Parameters
    ----------
    points : Tensor
        A coordinate Tensor, shape `B x S x D`
    latent : Tensor
        A Tensor of latent vectors, shape `B x N`

    Returns
    -------
    Tensor
        Each point concatenated to the respective latent vector, shape `B x S x (D + N)`
    """    
    B, S = points.shape[0], points.shape[1]
    z_exp = torch.stack([latent[i, :].expand(S, -1) for i in range(B)])
    x = torch.cat([points, z_exp], dim=-1).view(-1, points.shape[2] + latent.shape[1])
    return x

def label_to_interval(i: int, lo: float, hi: float, steps: int) -> float:
    """_summary_

    Parameters
    ----------
    i : int
        _description_
    lo : float
        _description_
    hi : float
        _description_
    steps : int
        _description_

    Returns
    -------
    float
        _description_
    """    
    return lo + (((hi - lo) / (steps - 1)) * i)