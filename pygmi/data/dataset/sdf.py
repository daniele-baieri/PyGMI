import random
import torch
from torch import Tensor
from typing import List, Dict, Tuple, Union, Any
from pygmi.data.dataset import MultiSourceData



class SDFUnsupervisedData(MultiSourceData):

    def __init__(
        self, 
        train_source_conf: List[Dict] = [], 
        test_source_conf: List[Dict] = [],
        preprocessing_conf: Dict = {},
        val_split: float = 0.0,
        batch_size: Dict[str, int] = {'train': 1, 'val': 1, 'test': 1},
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
        train_source_conf : List[Dict], optional
            List of configurations for multiple data sources. Each should specify a type
            (i.e. a subclass of ngt.data.sources.core.DataSource), and a configuration in 
            dict format depending on the source type (see ngt.data.sources), by default []
        test_source_conf : List[Dict], optional
            List of configurations for multiple data sources, by default []
        preprocessing_conf : Dict, optional
            Configuration for preprocessing procedure for the selected data sources, by default {}
        val_split : float, optional
            Fraction of training data serving for validation, by default 0.0
        batch_size : _type_, optional
            Batch size for train, test, val. Expects keys: {"train", "val", "test"}, 
            by default {'train': 1, 'val': 1, 'test': 1}
        surf_sample : int, optional
            Size of point sample representing a shape's surface, by default 16384
        global_space_sample : int, optional
            Size of global point samples in spatial sampling. Usually set equal to 
            `surf_sample // 8`. The final size of spatial samples is `surf_sample + global_space_sample`, 
            by default 2048
        global_sigma : float, optional
            Maximum coordinate of space for global spatial point sampling, by default 1.8
        local_sigma : float, optional
            std. dev. for local spatial point sampling; if None, preprocessed shapes 
            are expected to have key "mnfld_sigma", by default None
        use_normals : bool, optional
            Whether to sample normals together with surface points, by default True
        """          
        super(SDFUnsupervisedData, self).__init__(
            train_source_conf, test_source_conf, preprocessing_conf, batch_size, val_split)
        self.surf_sample = surf_sample
        self.space_sample = global_space_sample
        self.global_sigma = global_sigma
        self.use_normals = use_normals
        if local_sigma is None:
            self.fixed_local_sigma = False
        else:
            self.fixed_local_sigma = True
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

    def sample_surface(self, shape: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Samples a surface, optionally with normals and point-wise 
        local sampling standard deviations.

        Parameters
        ----------
        shape : Dict[str, Tensor]
            Preprocessed shape data. Expects keys {'surface', 'normals'} 
            and optionally 'mnfld_sigma'

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor]
            Surface sample, normals sample, sigmas sample. Normals and sigmas can be 
            None if they are not required by configuration
        """
        surf = shape['surface']
        indices = random.sample(range(surf.shape[0]), self.surf_sample)
        surf_sample = surf[indices, :]
        norm_sample = shape['normals'][indices, :] if self.use_normals else None
        sigmas_sample = shape['mnfld_sigma'][indices, :] if not self.fixed_local_sigma else None
        return surf_sample, norm_sample, sigmas_sample

    def collate(self, data: List[Dict], idxs: List[int]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Implementation of collate method. Loads a list of dictionaries with keys
        {'surface', 'normals', 'mnfld_sigma'} to a tuple of 4 Tensors (some of which may be
        None, depending on configuration).

        Parameters
        ----------
        data : List[Dict]
            A list of dictionaries with keys {'surface', 'normals', 'mnfld_sigma'} and Tensor values
        idxs : List[int]
            Indices of `data` in the dataset

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor, Tensor]
            `B x 1` LongTensor of indices of each shape in the batch,
            `B x S x 3` FloatTensor of surface samples for each shape in the batch,
            `B x S x 3` FloatTensor of normals for each sampled surface point (may be None),
            `B x T x 3` FloatTensor of space point samples for each shape in the batch
        """        
        shape_ids = torch.tensor(idxs, dtype=torch.long)
        samples = [self.sample_surface(x) for x in data]
        surf_sample = torch.stack([x[0] for x in samples])
        norm_sample = torch.stack([x[1] for x in samples]) if self.use_normals else None
        sigma = self.local_sigma if self.fixed_local_sigma else torch.stack([x[2] for x in samples])
        space_sample = self.sample_shape_space(surf_sample, sigma)
        return shape_ids, surf_sample, norm_sample, space_sample

    def load_data_point(self, path: str) -> Dict[str, Tensor]:
        """Implementation of load_data_point method. Loads a dictionary from `path` using
        `torch.load`.

        Parameters
        ----------
        path : str
            Path to data point stored on disk

        Returns
        -------
        Dict[str, Tensor]
            Loaded data point, must have keys {'surface', 'normals'} and optionally 'mnfld_sigma'
        """        
        return torch.load(path)



class SDFSupervisedData(MultiSourceData):

    def __init__(
        self, 
        train_source_conf: List[Dict] = [], 
        test_source_conf: List[Dict] = [],
        preprocessing_conf: Dict = {},
        val_split: float = 0.0,
        batch_size: Dict[str, int] = {'train': 1, 'val': 1, 'test': 1},
        surf_sample: int = 16384,
        space_sample: int = 16384,
        use_normals: bool = True
    ):
        """3D data for supervised (i.e. with ground truth distances) SDF tasks.

        Parameters
        ----------
        train_source_conf : List[Dict], optional
            List of configurations for multiple data sources. Each should specify a type
            (i.e. a subclass of ngt.data.sources.core.DataSource), and a configuration in 
            dict format depending on the source type (see ngt.data.sources), by default []
        test_source_conf : List[Dict], optional
            List of configurations for multiple data sources, by default []
        preprocessing_conf : Dict, optional
            Configuration for preprocessing procedure for the selected data sources, by default {}
        val_split : float, optional
            Fraction of training data serving for validation, by default 0.0
        batch_size : Dict[str, int], optional
            Batch size for train, test, val. Expects keys: {"train", "val", "test"}, 
            by default {'train': 1, 'val': 1, 'test': 1}
        surf_sample : int, optional
            Size of point sample representing a shape's surface, by default 16384
        space_sample : int, optional
            Size of point samples for a shape's embedding space, with distances, by default 16384
        use_normals : bool, optional
            Whether to sample normals together with surface points, by default True
        """                
        super(SDFSupervisedData, self).__init__(
            train_source_conf, test_source_conf, preprocessing_conf, batch_size, val_split)
        self.surf_sample = surf_sample
        self.space_sample = space_sample
        self.use_normals = use_normals

    def sample_distances(self, shape: Dict[str, Tensor]) -> Tensor:
        """Samples indices from a Tensor containing coordinates and 
        distance values, expected to be the value of `shape['dists']`

        Parameters
        ----------
        shape : Dict[str, Tensor]
            A dict with key 'dists' mapping to a `N x 4` Tensor

        Returns
        -------
        Tensor
            A sample of points and distances, with shape `self.space_sample x 4`
        """        
        dist = shape['dists']
        indices = random.sample(range(dist.shape[0]), self.space_sample)
        return dist[indices, :]

    def sample_surface(self, shape: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """Samples indices from a Tensor containing surface points of a shape, 
        expected to be the value of `shape['surface']`. If required by configuration,
        normals are sampled as well (from `shape['normals']`)

        Parameters
        ----------
        shape : Dict[str, Tensor]
            A dict with keys 'surface' mapping to a `N x 3` Tensor and 'normals' mapping
            to a `N x 3` Tensor (optional)

        Returns
        -------
        Tuple[Tensor, Tensor]
            Surface sample, normals sample. Normals can be 
            None if they are not required by configuration
        """        
        surf = shape['surface']
        indices = random.sample(range(surf.shape[0]), self.surf_sample)
        surf_sample = surf[indices, :]
        norm_sample = shape['normals'][indices, :] if self.use_normals else None
        return surf_sample, norm_sample

    def collate(self, data: List[Dict], idxs: List[int]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Implementation of collate method. Loads a list of dictionaries with keys
        {'surface', 'normals', 'dists'} to a tuple of 4 Tensors (some of which may be
        None, depending on configuration).

        Parameters
        ----------
        data : List[Dict]
            A list of dictionaries with keys {'surface', 'normals', 'dists'} and Tensor values
        idxs : List[int]
            Indices of `data` in the dataset

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor, Tensor]
            `B x 1` LongTensor of indices of each shape in the batch,
            `B x S x 3` FloatTensor of surface samples for each shape in the batch,
            `B x S x 3` FloatTensor of normals for each sampled surface point (may be None),
            `B x T x 3` FloatTensor of space point samples for each shape in the batch
        """        
        shape_ids = torch.tensor(idxs, dtype=torch.long)
        samples = [self.sample_surface(x) for x in data]
        surf_sample = torch.stack([x[0] for x in samples])
        norm_sample = torch.stack([x[1] for x in samples]) if self.use_normals else None
        dist_sample = torch.stack([self.sample_distances(x) for x in data])
        return shape_ids, surf_sample, norm_sample, dist_sample

    def load_data_point(self, path: str) -> Dict[str, Tensor]:
        """Implementation of load_data_point method. Loads a dictionary from `path` using
        `torch.load`.

        Parameters
        ----------
        path : str
            Path to data point stored on disk

        Returns
        -------
        Dict[str, Tensor]
            Loaded data point, must have keys {'surface', 'normals', 'dists'}
        """        
        return torch.load(path)