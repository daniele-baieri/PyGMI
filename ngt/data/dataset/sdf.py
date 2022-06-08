import random
import torch
import ngt.data.sources
import ngt.data.preprocess
from torch import Tensor
from typing import List, Dict, Tuple, Union
from ngt.data.dataset import MultiSourceData



class SDFUnsupervisedData(MultiSourceData):

    def __init__(
        self, 
        train_source_conf: List[Dict] = [], 
        test_source_conf: List[Dict] = [],
        preprocessing_conf: Dict = {},
        val_split: float = 0.0,
        batch_size: Dict[str, int] = {'train': 0, 'val': 0, 'test': 0},
        surf_sample: int = 16384,
        global_space_sample: int = 2048,
        global_sigma: float = 1.8,
        local_sigma: float = None,
        use_normals: bool = True
    ):
        """3D data for unsupervised (i.e. without ground truth distances)
        SDF tasks. Uses supersampled meshes with normals, and by default
        uses point-wise standard deviations for spatial sampling.

        Parameters
        ----------
        train_source_conf : List[Dict]
            List of configurations for multiple data sources. Each should specify a type
            (i.e. a subclass of ngt.data.sources.core.DataSource), and a configuration in 
            dict format depending on the source type (see ngt.data.sources)
        test_source_conf : List[Dict]
            List of configurations for multiple data sources
        preprocessing_conf : Dict
            Configuration for preprocessing procedure for the selected data sources
        batch_size : Dict[str, int]
            Batch size for train, test, val. Expects keys: {"train", "val", "test"}
        val_split : float
            Fraction of training data serving for validation
        surf_sample : int
            Size of point sample representing a shape's surface
        global_space_sample : int
            Size of global point samples in spatial sampling. Usually set equal to 
            `surf_sample // 8`. The final size of spatial samples is `surf_sample + global_space_sample`
        global_sigma : float
            Maximum coordinate of space for global spatial point sampling
        local_sigma : float, optional
            std. dev. for local spatial point sampling; if None, preprocessed shapes 
            are expected to have key "mnfld_sigma", by default None
        use_normals : bool, optional
            Whether to sample normals together with surface points, by default True
        """        
        super(SDFUnsupervisedData, self).__init__(
            train_source_conf, test_source_conf, preprocessing_conf, batch_size, val_split)
        self.preproc_fn = ngt.data.preprocess.upsample_with_normals
        self.surf_sample = surf_sample
        self.space_sample = global_space_sample
        self.global_sigma = global_sigma
        self.use_normals = use_normals
        if local_sigma is None:
            self.use_local_sigma = False
        else:
            self.use_local_sigma = True
            self.local_sigma = local_sigma

    def sample_shape_space(self, point_cloud: Tensor, local_sigma: Union[Tensor, float]) -> Tensor:
        """Samples points from embedding space, concatenating a small uniformly sampled
        (global) sample with a large Gaussian local sample, computed either with point-wise
        standard deviations (if `type(local_sigma) == Tensor`) or fixed standard deviation
        (if `type(local_sigma) == float`)

        Parameters
        ----------
        point_cloud : Tensor
            Surface samples of shapes for which to perform spatial sampling. Shape: `B x S x 3`
        local_sigma : Union[Tensor, float]
            Standard deviation for local sampling. Either fixed (if type is float) or point-wise
            (if type is Tensor)

        Returns
        -------
        Tensor
            A random sample of points around each given point cloud
        """        
        sample_local = point_cloud + (torch.randn_like(point_cloud) * local_sigma)
        sample_global = (
            2 * self.global_sigma * torch.rand(
                point_cloud.shape[0], self.space_sample, point_cloud.shape[2]
            )) - self.global_sigma
        return torch.cat([sample_local, sample_global], dim=1)

    def sample_surface(self, shape: Dict[str, Tensor]) -> List[Tensor]:
        """Samples a surface, optionally with normals and point-wise 
        local sampling standard deviations.

        Parameters
        ----------
        shape : Dict[str, Tensor]
            Preprocessed shape data. Expects keys {"surface", "normals", 
            "vertices", "faces"} and optionally "mnfld_sigma"

        Returns
        -------
        List[Tensor]
            List of three elements: surface sample, normals sample, sigmas sample.
            Normals and sigmas can be None if they are not required (by configuration)
        """
        surf = shape['surface']
        indices = random.sample(range(surf.shape[0]), self.surf_sample)
        out = [surf[indices, :]]
        out.append(shape['normals'][indices, :] if self.use_normals else None)
        out.append(shape['mnfld_sigma'][indices, :] if not self.use_local_sigma else None)
        return out

    def collate(self, paths: List[Tuple[str, int]]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        shape_ids = torch.tensor([x[1] for x in paths], dtype=torch.float)
        data = [[torch.load(p[0]) for p in paths]]
        samples = [self.sample_surface(shape) for shape in data]
        surf_sample = torch.stack([x[0] for x in samples])
        norm_sample = torch.stack([x[1] for x in samples]) if self.use_normals else None
        sigma = self.local_sigma if self.use_local_sigma else torch.stack([x[2] for x in samples]).view(-1, -1, 1)
        space_sample = self.sample_shape_space(surf_sample, sigma)
        return shape_ids, surf_sample, norm_sample, space_sample


class SDFSupervisedData(MultiSourceData):

    def __init__(
        self, 
        train_source_conf: List[Dict] = [], 
        test_source_conf: List[Dict] = [],
        preprocessing_conf: Dict = {},
        val_split: float = 0.0,
        batch_size: Dict[str, int] = {'train': 0, 'val': 0, 'test': 0},
        surf_sample: int = 16384,
        space_sample: int = 16384,
        use_normals: bool = True
    ):
        """3D data for supervised (i.e. with ground truth distances) SDF tasks.

        Parameters
        ----------
        train_source_conf : List[Dict]
            List of configurations for multiple data sources. Each should specify a type
            (i.e. a subclass of ngt.data.sources.core.DataSource), and a configuration in 
            dict format depending on the source type (see ngt.data.sources)
        test_source_conf : List[Dict]
            List of configurations for multiple data sources
        preprocessing_conf : Dict
            Configuration for preprocessing procedure for the selected data sources
        batch_size : Dict[str, int]
            Batch size for train, test, val. Expects keys: {"train", "val", "test"}
        val_split : float
            Fraction of training data serving for validation
        surf_sample : int
            Size of point sample representing a shape's surface
        space_sample : int
            Size of point samples for a shape's embedding space, with distances
        use_normals : bool, optional
            Whether to sample normals together with surface points, by default True
        """          
        super(SDFSupervisedData, self).__init__(
            train_source_conf, test_source_conf, preprocessing_conf, batch_size, val_split)
        self.surf_sample = surf_sample
        self.space_sample = space_sample
        self.use_normals = use_normals

    def sample_distances(self, shape: Dict[str, Tensor]) -> Tensor:
        dist = shape['dists']
        indices = random.sample(range(dist.shape[0]), self.space_sample)
        return dist[indices, :]

    def sample_surface(self, shape: Dict[str, Tensor]) -> List[Tensor]:
        surf = shape['surface']
        indices = random.sample(range(surf.shape[0]), self.surf_sample)
        out = [surf[indices, :]]
        out.append(shape['normals'][indices, :] if self.use_normals else None)
        return out

    def collate(self, paths: List[Tuple[str, int]]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        shape_ids = torch.tensor([x[1] for x in paths], dtype=torch.float)
        data = [[torch.load(p[0]) for p in paths]]
        samples = [self.sample_surface(shape) for shape in data]
        surf_sample = torch.stack([x[0] for x in samples])
        norm_sample = torch.stack([x[1] for x in samples]) if self.use_normals else None
        dist_sample = torch.stack([self.sample_distances(shape) for shape in data])
        return shape_ids, surf_sample, norm_sample, dist_sample
