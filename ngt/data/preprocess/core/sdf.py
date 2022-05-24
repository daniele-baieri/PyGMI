import trimesh
import os
import torch
import numpy as np
from torch_geometric.data import Data as PyGData
from scipy.spatial import cKDTree
from CGAL.CGAL_Kernel import Triangle_3, Point_3
from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup


def process_shape(
    S: PyGData,
    sample: int, 
    out_path: str
) -> None: 

    # Create trimesh # 
    F = S.face
    V = S.pos
    mesh = trimesh.Trimesh(vertices=V, faces=F.T)

    # Sample points and normals #
    pnts, face_index = trimesh.sample.sample_surface(mesh, sample)
    center = np.mean(pnts, axis=0)
    pnts = pnts - np.expand_dims(center, axis=0)
    normals = mesh.face_normals[face_index]
    
    # Normalize registered surface points #
    V = V[:, [0, 2, 1]] - torch.from_numpy(center[[0, 2, 1]]).unsqueeze(0).float()
    area = np.sqrt(mesh.area)
    pnts /= area
    V /= area

    # Instantiate CGAL AABB tree #
    triangles = []
    for tri in mesh.triangles:
        T = (tri - center) / area
        a = Point_3(T[0][0], T[0][1], T[0][2])  # (tri[0][0] - center[0]), (tri[0][1] - center[1]), (tri[0][2] - center[2]))
        b = Point_3(T[1][0], T[1][1], T[1][2])  # (tri[1][0] - center[0]), (tri[1][1] - center[1]), (tri[1][2] - center[2]))
        c = Point_3(T[2][0], T[2][1], T[2][2])  # (tri[2][0] - center[0]), (tri[2][1] - center[1]), (tri[2][2] - center[2]))
        triangles.append(Triangle_3(a, b, c))
    tree = AABB_tree_Triangle_3_soup(triangles)

    # Sample points #
    sigmas = []
    ptree = cKDTree(pnts)
    for p in np.array_split(pnts,100,axis=0):
        d = ptree.query(p,51)
        sigmas.append(d[0][:,-1])

    sigmas = np.concatenate(sigmas)
    sigmas_big = 0.2 * np.ones_like(sigmas)

    sample = np.concatenate([
        pnts + np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pnts.shape),
        pnts + np.expand_dims(sigmas_big,-1) * np.random.normal(0.0,1.0, size=pnts.shape)], axis=0)

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
            'surface': torch.from_numpy(pnts)[:, [0, 2, 1]].float(),
            'dists': sample_dists,
            'matching': V,
            'faces': F.T,
            'normals': torch.from_numpy(normals)[:, [0, 2, 1]].float()
        },
        out_path
    )
