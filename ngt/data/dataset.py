import re
import random
import numpy as np
import torch
import pytorch_lightning as pl
import ngt.data.sources
import ngt.data.preprocess
from torch.utils.data import DataLoader
from torch import Tensor
from typing import Optional, List, Dict, Tuple, Any, Union
from ngt.data.preprocess import gather_fnames, process_source
from ngt.utils.files import validate_fnames



class PathList:

    def __init__(self, l: list):
        self.data = l

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        return self.data[idx], idx

    def __len__(self) -> int:
        return len(self.data)

    def append(self, x: Any) -> None:
        self.data.append(x)

    def union(self, x: List[Any]) -> None:
        self.data += x



class MultiSourceData(pl.LightningDataModule):

    def __init__(
        self, 
        train_source_conf: List[Dict], 
        test_source_conf: List[Dict],
        preprocessing_conf: Dict,
        batch_size: Dict[str, int],
        val_split: float
    ):
        """Generic multi-source data module. Should be subclassed to define
        custom behaviour for collecting preprocessed data into batches.

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
        """
        super(MultiSourceData, self).__init__()
        self.train = train_source_conf
        self.test = test_source_conf
        self.preproc_conf = preprocessing_conf
        self.preproc_fn = getattr(ngt.data.preprocess, self.preproc_conf['script'])
        self.train_paths = PathList([])
        self.val_paths = PathList([])
        self.test_paths = PathList([])
        self.val_split = val_split
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        
        if stage == 'fit' or stage is None:
            for conf in self.train:
                source = getattr(ngt.data.sources, conf['type'])(**conf['source_conf'])
                sname = '_'.join(re.split('/|\\', conf['source_conf']['source'].replace('/', '_')))
                fnames = sorted(gather_fnames(self.preproc_conf['out_dir'], sname, len(source)))
                if self.preproc_conf['do_preprocessing']:
                    process_source(source, self.preproc_fn, fnames, self.preproc_conf['conf'])
                else:
                    if not validate_fnames(fnames):
                        raise RuntimeError('Processed files missing with preprocessing disabled.')
                self.train_paths.union(fnames)

            n_val = int(len(self.train_paths) * self.val_split)
            val_idxs = random.sample(range(len(self.train_paths)), n_val)
            self.val_paths.union([self.train_paths[i] for i in val_idxs])
            self.train_paths = np.delete(self.train_paths, val_idxs)
                
        if stage == 'test' or stage is None:  
            for conf in self.test:
                source = getattr(ngt.data.sources, conf['type'])(**conf['source_conf'])
                sname = '_'.join(re.split('/|\\', conf['source_conf']['source'].replace('/', '_')))
                fnames = sorted(gather_fnames(self.preproc_conf['out_dir'], sname, len(source)))
                if self.preproc_conf['do_preprocessing']:
                    process_source(source, self.preproc_fn, fnames, self.preproc_conf['conf'])
                else:
                    if not validate_fnames(fnames):
                        raise RuntimeError('Processed files missing with preprocessing disabled.')
                self.train_paths.union(fnames)

    def collate(self, paths: List[Tuple[str, int]]) -> Any:
        """Users should override this method to define what happens with stored 
        preprocessed data when it is required for usage.

        Parameters
        ----------
        paths : List[str]
            List of paths to preprocessed data

        Returns
        -------
        Any
            Data points ready for usage in NGT pipelines (this will be returned
            when iterating on the DataLoader)

        Raises
        ------
        NotImplementedError
            Has to be implemented in subclasses
        """
        raise NotImplementedError()
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_paths, self.batch_size['train'], shuffle=True, collate_fn=self.collate)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_paths, self.batch_size['val'], collate_fn=self.collate)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_paths, self.batch_size['test'], collate_fn=self.collate)


class SDFUnsupervisedData(MultiSourceData):

    def __init__(
        self, 
        train_source_conf: List[Dict], 
        test_source_conf: List[Dict],
        preprocessing_conf: Dict,
        val_split: float,
        batch_size: Dict[str, int],
        surf_sample: int,
        global_space_sample: int,
        global_sigma: float,
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
            Size of global point samples in spatial sampling. Usually set to 
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
                point_cloud.shape[0], self.space_sample // 8, point_cloud.shape[2]
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
        train_source_conf: List[Dict], 
        test_source_conf: List[Dict],
        preprocessing_conf: Dict,
        val_split: float,
        batch_size: Dict[str, int],
        surf_sample: int,
        space_sample: int,
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
