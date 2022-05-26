import trimesh
import torch
import numpy as np
from typing import Tuple
from torch import Tensor
from torch_geometric.data import Data as PyGData
from scipy.spatial import cKDTree
from CGAL.CGAL_Kernel import Triangle_3, Point_3
from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup



def _upsample_and_normalize(
    S: PyGData, sample: int
) -> Tuple[trimesh.Trimesh, Tensor, Tensor, Tensor, np.ndarray, float]:

    # Create trimesh # 
    F = S.face
    V = S.pos
    mesh = trimesh.Trimesh(vertices=V, faces=F.T)

    # Sample points and normals #
    pnts, face_index = trimesh.sample.sample_surface(mesh, sample)
    center = np.mean(pnts, axis=0)
    pnts = pnts - np.expand_dims(center, axis=0)
    normals = torch.from_numpy(mesh.face_normals[face_index])[:, [0, 2, 1]].float()
    
    # Normalize registered surface points and upsample #
    V = V[:, [0, 2, 1]] - torch.from_numpy(center[[0, 2, 1]]).unsqueeze(0).float()
    area = np.sqrt(mesh.area)
    pnts /= area
    V /= area

    return mesh, V, torch.from_numpy(pnts)[:, [0, 2, 1]].float(), normals, center, area

def _compute_sigmas(pnts: Tensor) -> np.ndarray:

    query = pnts.numpy()
    sigmas = []
    ptree = cKDTree(query)
    for p in np.array_split(query,100,axis=0):
        d = ptree.query(p, 51)  # sigma = dist from 50-th NN (heuristic)
        sigmas.append(d[0][:,-1])

    return np.concatenate(sigmas)


def get_distance_values(S: PyGData, out_path: str, sample: int, global_sigma: float = 0.2) -> None: 
    """Preprocess a 3D shape for SDF tasks: get ground truth distance values
    without sign. Useful for learning SDFs using (e.g.) sign agnostic regression.

    Parameters
    ----------
    S : PyGData
        A torch_geometric.data.Data object, representing a mesh
    sample : int
        Surface sample size and (distance sample size) / 2
    out_path : str
        Memory location to save preprocessed data
    global_sigma : float, optional
        _description_, by default 0.2
    """    
        
    mesh, V, pnts, normals, center, area = _upsample_and_normalize(S, sample)

    # Instantiate CGAL AABB tree #
    triangles = []
    for tri in mesh.triangles:
        T = (tri - center) / area
        a = Point_3(T[0][0], T[0][1], T[0][2])  # (tri[0][0] - center[0]), (tri[0][1] - center[1]), (tri[0][2] - center[2]))
        b = Point_3(T[1][0], T[1][1], T[1][2])  # (tri[1][0] - center[0]), (tri[1][1] - center[1]), (tri[1][2] - center[2]))
        c = Point_3(T[2][0], T[2][1], T[2][2])  # (tri[2][0] - center[0]), (tri[2][1] - center[1]), (tri[2][2] - center[2]))
        triangles.append(Triangle_3(a, b, c))
    tree = AABB_tree_Triangle_3_soup(triangles)

    # Sample points with 50-th NN heuristic #
    sigmas = _compute_sigmas(pnts)
    sigmas_big = global_sigma * np.ones_like(sigmas)

    sample = np.concatenate([
        pnts + np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pnts.shape),
        pnts + np.expand_dims(sigmas_big,-1) * np.random.normal(0.0, 1.0, size=pnts.shape)], axis=0)

    # Compute distances # 
    dists = []
    for np_query in sample:  
        cgal_query = Point_3(np_query[0].astype(np.double), np_query[1].astype(np.double), np_query[2].astype(np.double))

        cp = tree.closest_point(cgal_query)
        cp = np.array([cp.x(), cp.y(), cp.z()])
        dist = np.sqrt(((cp - np_query)**2).sum(axis=0))

        dists.append(dist)
    dists = np.array(dists).reshape(-1, 1)

    sample_dists = torch.from_numpy(np.concatenate([sample, dists], axis=-1))[:, [0, 2, 1, 3]].float()

    # Save everything to pth #
    torch.save(
        {
            'surface': pnts,
            'dists': sample_dists,
            'vertices': V,
            'faces': S.face,
            'normals': normals
        },
        out_path
    )

def upsample_with_normals(
    S: PyGData,
    out_path: str,
    sample: int,
    mnfld_sigma: bool = False
) -> None: 
    """Preprocess a 3D shape for SDF tasks: upsample mesh vertices and 
    normals, optionally compute spatial sampling sigmas

    Parameters
    ----------
    S : PyGData
        A torch_geometric.data.Data object, representing a mesh
    sample : int
        Surface (and normals) sample size
    out_path : str
        Memory location to save preprocessed data
    mnfld_sigma: bool
        Specifies whether to compute space sampling std for each point
    """    
    
    _, V, pnts, normals, _, _ = _upsample_and_normalize(S, sample)

    sigmas = None
    if mnfld_sigma:
        sigmas = torch.from_numpy(_compute_sigmas(pnts)).float()
        
    # Save everything to pth #
    torch.save(
        {
            'surface': pnts,
            'vertices': V,
            'faces': S.face,
            'normals': normals,
            'mnfld_sigma': sigmas
        },
        out_path
    )

def center_point_cloud(S: PyGData, out_path: str) -> None:
    """Simply save a point cloud to disk, after centering in the origin

    Parameters
    ----------
    S : PyGData
        A torch_geometric.data.Data object, representing a point cloud (all 
        attributes are ignored except for S.pos)
    """    

    # Center in origin
    V = S.pos - S.pos.mean(dim=0, keepdim=True)

    # Save everything to pth #
    torch.save({'surface': V}, out_path)