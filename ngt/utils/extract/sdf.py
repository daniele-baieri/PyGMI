import torch
import numpy as np
import ngt.utils as utils
from torch import Tensor
from skimage import measure
from typing import Tuple, Callable, Literal


def triangulate_sdf(
    f: Callable,  
    dim: int, 
    res: int, 
    latent: Tensor = None,
    bound: float = 1.0, 
    device: Literal['cpu', 'cuda'] = 'cpu', 
    level: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts a mesh from an SDF using the Marching Cubes algorithm.

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
    latent : Tensor, optional
        Latent vectors representing shapes in the latent space of `f`.
        If None, `f` is assumed to be non-parametric, by default None
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
    volume = evaluate_sdf_grid(f, dim, res, bound, device, latent)
    verts, faces = marching_cubes(volume, (2 * bound) / (res - 1), level)
    if len(faces) > 1:
        verts -= bound
    return verts, faces

def evaluate_sdf_grid(
    f: Callable, 
    dim: int, 
    res: int, 
    bound: float, 
    device: str, 
    latent: Tensor = None
) -> np.ndarray:
    """Evaluates an SDF on a regular voxel grid.

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
    latent : Tensor, optional
        Latent vectors representing shapes in the latent space of `f`.
        If None, `f` is assumed to be non-parametric, by default None

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
            if latent is not None:
                volume.append(f(pnts.unsqueeze(0), latent).detach().cpu().numpy())
            else:
                volume.append(f(pnts).detach().cpu().numpy())
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



'''
    volume = []
    with torch.no_grad():
        G = utils.make_grid(res, bound, dim)
        split = torch.split(G, 100000, dim=0)
        for j in range(len(split)):
            pnts = split[j].to(device)
            if latent is not None:
                volume.append(f(pnts.unsqueeze(0), latent).detach().cpu().numpy())
            else:
                volume.append(f(pnts).detach().cpu().numpy())
            
        volume = np.concatenate(volume, axis=0)
        try:
            verts, faces, _, _ = measure.marching_cubes(
                volume.reshape(res, res, res).transpose([1, 0, 2]),
                level=level, spacing=[(2 * bound) / (res - 1)] * 3
            )
        except:
            verts = np.array([[bound * 3]] * dim)
            faces = np.array([[0, 1, 2]])
    return verts - bound, faces'''