import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch import Tensor
from typing import List, Tuple



class Autodecoder(nn.Module):

    def __init__(self, num_data_pts: int, latent_dim: int, sigma: float = 0.0):
        """Creates an Autodecoder, as proposed in https://arxiv.org/abs/1901.05103.

        Parameters
        ----------
        num_data_pts : int
            Number of data points to represent (= # latent vectors)
        latent_dim : int
            Dimensionality of optimized latent vectors
        sigma : float
            stddev of initial distribution
        """        
        super(Autodecoder, self).__init__()
        self.num_vectors = num_data_pts
        self.latent_dim = latent_dim
        self.init_sigma = sigma
        self.vectors = nn.parameter.Parameter(
            torch.randn((num_data_pts, latent_dim)) * sigma)

    def forward(self, idx: Tensor) -> Tensor:
        return self.vectors[idx, :]


### PointNet++ ###

class PointNet2Layer(nn.Module):

    def __init__(self, point_filter: nn.Module, radius: float, density: float, set_filter: nn.Module = None):
        """A PointNet++ layer, performing radius aggregation over a 
        percentage of input points.

        Parameters
        ----------
        point_filter : nn.Module
            Point-wise MLP processing point features and coordinates
        radius : float
            Maximum distance of neighbors of each pivot point
        density : float
            Fraction of pivot points over given input point samples
        set_filter : nn.Module, optional
            Point-wise MLP processing point features and coordinates after aggregation, by default None
        """        
        super(PointNet2Layer, self).__init__()

        self.rad = radius
        self.density = density
        self.conv = gnn.PointNetConv(point_filter, set_filter, False)
        self.do_set_filter = set_filter is not None

    def forward(self, pos: Tensor, batch: Tensor, x: Tensor = None) -> Tuple[Tensor]:
        """Performs PointNet++ convolution over input.

        Parameters
        ----------
        pos : Tensor
            Point coordinates, shape `N x D`
        batch : Tensor
            Batch tensor, shape `N x 1`, batch size = `max(batch)`
        x : Tensor, optional
            Point features, shape `N x F`

        Returns
        -------
        Tuple[Tensor]
            Pivot points coordinates, processed features, and batch tensor
        """        
        centr_idx = gnn.fps(pos, batch, ratio=self.density)
        centroids, centr_batch = pos[centr_idx], batch[centr_idx]
        row, col = gnn.radius(pos, centroids, self.rad, batch, centr_batch, max_num_neighbors=64)
        centr_x = None if x is None else x[centr_idx]
        out = self.conv((x, centr_x), (pos, centroids), torch.stack([col, row]))
        return centroids, out, centr_batch


class PointNet2Encoder(nn.Module):

    def __init__(
        self, 
        density: List[float], 
        radius: List[float], 
        pf_size: List[List[int]], 
        sf_size: List[List[int]] = None
    ):
        """Creates a PointNet++ encoder, mapping point clouds to n-dimensional vectors.

        Parameters
        ----------
        density : List[float]
            Fraction of pivot points over input sample for each layer
        radius : List[float]
            Maximum distance of neighbors of each pivot point for each layer
        pf_size : List[List[int]]
            Layer dimensions of MLP point filters for each layers
        sf_size : List[List[int]], optional
            Layer dimensions of MLP set filters for each layers, by default None
        """        
        super(PointNet2Encoder, self).__init__()

        num_layers = len(density)
        self.layers = nn.ModuleList()
        do_set_filtering = sf_size is not None
        for l in range(num_layers):
            pf = gnn.MLP(pf_size[l])
            sf = gnn.MLP(sf_size[l]) if do_set_filtering else None
            layer = PointNet2Layer(pf, radius[l], density[l], sf)
            self.layers.append(layer)

    def forward(self, pos: Tensor, batch: Tensor, x: Tensor = None) -> Tensor:
        """Encodes a batch of point clouds.

        Parameters
        ----------
        pos : Tensor
            Point coordinates, shape `N x D`
        batch : Tensor
            Batch tensor, shape `N x 1`, batch size = `max(batch)`
        x : Tensor, optional
            Point features, shape `N x F`

        Returns
        -------
        Tensor
            Latent vectors for each input point cloud, shape `N x L`
        """      
        p, b, x = pos, batch, x
        for layer in self.layers:
            p, x, b = layer(p, b, x)
        z = gnn.global_max_pool(x, b)
        return z