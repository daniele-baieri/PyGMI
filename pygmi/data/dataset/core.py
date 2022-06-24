import re
import random
import numpy as np
import pytorch_lightning as pl
import pygmi.data.sources
import pygmi.data.preprocess
from torch.utils.data import DataLoader
from typing import Optional, List, Dict, Tuple, Any, Collection
from pygmi.data.preprocess import gather_fnames, process_source
from pygmi.utils.files import validate_fnames, mkdir_ifnotexists




class _ListWithIndices(list):

    def __init__(self):
        """Initializes a subclass of list which returns objects and their indices.
        """        
        super(_ListWithIndices, self).__init__()

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """Overrides the List __getitem__ to return the index along with the object.

        Parameters
        ----------
        idx : int
            Index of objects to retrieve

        Returns
        -------
        Tuple[Any, int]
            Retrieved object and index
        """        
        return super(_ListWithIndices, self).__getitem__(idx), idx    



class MultiSourceData(pl.LightningDataModule):

    def __init__(
        self, 
        train_source_conf: List[Dict] = [], 
        test_source_conf: List[Dict] = [],
        preprocessing_conf: Dict = {},
        batch_size: Dict[str, int] = {'train': 1, 'val': 1, 'test': 1},
        val_split: float = 0.0
    ):      
        """Generic multi-source data module. Should be subclassed to define
        custom behaviour for collecting preprocessed data into batches.

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
        batch_size : Dict[str, int], optional
            Batch size for train, test, val. Expects keys: {"train", "val", "test"}, 
            by default {'train': 1, 'val': 1, 'test': 1}
        val_split : float, optional
            Fraction of training data serving for validation, by default 0.0
        """
        super(MultiSourceData, self).__init__()
        self.train = train_source_conf
        self.test = test_source_conf
        self.preproc_conf = preprocessing_conf
        self.preproc_fn = getattr(pygmi.data.preprocess, self.preproc_conf['script'])
        self.train_paths = _ListWithIndices() # PathList([])
        self.val_paths = _ListWithIndices() # PathList([])
        self.test_paths = _ListWithIndices() # PathList([])
        self.val_split = val_split
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        
        if stage == 'fit' or stage is None:
            for conf in self.train:
                source = getattr(pygmi.data.sources, conf['type'])(**conf['source_conf'])
                sname = '_'.join(re.split('/|\\\\', conf['source_conf']['source']))
                fnames = sorted(gather_fnames(self.preproc_conf['out_dir'], sname, len(source)))
                mkdir_ifnotexists(self.preproc_conf['out_dir'])
                if self.preproc_conf['do_preprocessing']:
                    process_source(source, fnames, self.preproc_fn, self.preproc_conf['conf'])
                else:
                    if not validate_fnames(fnames):
                        raise RuntimeError('Processed files missing with preprocessing disabled.')
                self.train_paths += fnames

            n_val = int(len(self.train_paths) * self.val_split)
            val_idxs = random.sample(range(len(self.train_paths)), n_val)
            self.val_paths += [self.train_paths[i] for i in val_idxs]
            for i in val_idxs:
                self.train_paths.pop(i)
                
        if stage == 'test' or stage is None:  
            for conf in self.test:
                source = getattr(pygmi.data.sources, conf['type'])(**conf['source_conf'])
                sname = '_'.join(re.split('/|\\', conf['source_conf']['source'].replace('/', '_')))
                fnames = sorted(gather_fnames(self.preproc_conf['out_dir'], sname, len(source)))
                if self.preproc_conf['do_preprocessing']:
                    process_source(source, self.preproc_fn, fnames, self.preproc_conf['conf'])
                else:
                    if not validate_fnames(fnames):
                        raise RuntimeError('Processed files missing with preprocessing disabled.')
                self.train_paths += fnames

    def collate(self, data: List[Any], idxs: List[int]) -> Any:
        """Users should override this method to define how to put
        several data points in batch form.

        Parameters
        ----------
        data : List[Any]
            List of data points and their indices in the dataset
        idxs : List[int]
            Indices of `data` in the dataset

        Returns
        -------
        Any
            Data points in batch form, ready for usage in train/val/test loops 
            (will be returned when iterating on the DataLoader)

        Raises
        ------
        NotImplementedError
            Has to be implemented in subclasses
        """
        raise NotImplementedError()

    def load_data_point(self, path: str) -> Any:
        """Users should override this method to define how data points
        are loaded from disk into Python objects.

        Parameters
        ----------
        path : str
            A path to a data point stored on disk.

        Returns
        -------
        Any
            The Python object storing the loaded data point

        Raises
        ------
        NotImplementedError
            Has to be implemented in subclasses
        """        
        raise NotImplementedError()

    def load_and_collate(self, paths: List[Tuple[str, int]]) -> Any:
        """Collate function for DataLoaders, useful for datasets stored on
        disk and accessed on demand. Loads each given path and returns the
        batch of data points.

        Parameters
        ----------
        paths : List[Tuple[str, int]]
            A list of filepaths and the indices in the dataset of their corresponding
            data points

        Returns
        -------
        Any
            Data points in batch form, ready for usage in train/val/test loops 
            (will be returned when iterating on the DataLoader)

        """        
        loaded = [self.load_data_point(p[0]) for p in paths]
        collated = self.collate(loaded, [p[1] for p in paths])
        return collated
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_paths, self.batch_size['train'], shuffle=True, collate_fn=self.load_and_collate)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_paths, self.batch_size['val'], collate_fn=self.load_and_collate)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_paths, self.batch_size['test'], collate_fn=self.load_and_collate)