import torch
import numpy as np
import ngt.utils as utils
from skimage import measure
from typing import Tuple, Callable, Literal


def extract_level_set(
    f: Callable,  
    dim: int, 
    res: int, 
    bound: float = 1.0, 
    device: Literal['cpu', 'cuda'] = 'cpu', 
    level: float = 0.0,
    *f_args, **f_kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Approximates a given level set of an implicit function with a mesh.
    The function is evaluated on a regular voxel grid and the output mesh 
    is extracted using the Marching Cubes algorithm.

    Parameters
    ----------
    f : Callable
        Callable representing the implicit function. Its first argument
        has to be a tensor of spatial coordinates of shape (B, S, dim)
    dim : int
        Dimensionality of query points for `f`
    res : int
        Grid resolution for mesh extraction
    bound : float, optional
        Maximum coordinate of voxel grid, by default 1.0
    device : Literal['cpu', 'cuda'], optional
        Device on which to run the computation, by default 'cpu'
    level : float, optional
        Which level set to extract, by default 0.0

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Vertices and faces of extracted mesh. If there is no zero crossing, 
        it returns a single triangle collapsing on the origin.
    """    
    volume = grid_evaluation(f, dim, res, bound, device, f_args, f_kwargs)
    verts, faces = marching_cubes(volume, (2 * bound) / (res - 1), level)
    if len(faces) > 1:
        verts -= bound
    return verts, faces

def grid_evaluation(
    f: Callable, 
    dim: int, 
    res: int, 
    bound: float, 
    device: str, 
    *f_args, **f_kwargs,
) -> np.ndarray:  
    """Evaluates an implicit function on a regular voxel grid. Input space
    coordinates be given as a tensor with shape `(B, S, dim)` where
    `B` is the batch size and `S` is the sample size (number of points).
    args and kwargs are forwarded to f when it is invoked.

    Parameters
    ----------
    f : Callable
        SDF function. If parametric (i.e. `latent is not None`), it expects 
        two Tensors of shapes `B x S x dim` and `B x n`. Otherwise, it expecst
        one Tensor of shape `B x S x dim`
    dim : int
        Dimensionality of query points for `f`
    res : int
        Grid resolution for mesh extraction
    bound : float
        Maximum coordinate of voxel grid, by default 1.0
    device : str
        Device on which to run the computation, by default 'cpu'

    Returns
    -------
    np.ndarray
        Shape `res x res x res`, containing SDF values
    """    
    
    volume = []
    with torch.no_grad():
        G = utils.make_grid(res, bound, dim)
        split = torch.split(G, 100000, dim=0)
        for j in range(len(split)):
            pnts = split[j].to(device)
            volume.append(f(pnts.unsqueeze(0), *f_args, **f_kwargs).detach().cpu().numpy())
        return np.concatenate(volume, axis=0).reshape(res, res, res).transpose([1, 0, 2])

def marching_cubes(volume: np.ndarray, voxel_size: float, level: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Invokes Marching Cubes on a voxel grid containing a scalar function.
    Gracefully handles the case of functions with no level crossing.

    Parameters
    ----------
    volume : np.ndarray
        Shape `N x N x N`
    voxel_size : float
        Size of a voxel in all dimensions
    level : float, optional
        Function level set to extract, by default 0.0

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Vertices and faces of extracted mesh. If there is no zero crossing, 
        it returns a single triangle collapsing on the origin.
    """    
    try:
        verts, faces, _, _ = measure.marching_cubes(volume, level=level, spacing=[voxel_size] * 3)  # [(2 * bound) / (res - 1)] * 3
    except:
        verts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        faces = np.array([[0, 1, 2]])  
    return verts, faces
