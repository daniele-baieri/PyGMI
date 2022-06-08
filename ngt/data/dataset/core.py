import re
import random
import numpy as np
import pytorch_lightning as pl
import ngt.data.sources
import ngt.data.preprocess
from torch.utils.data import DataLoader
from typing import Optional, List, Dict, Tuple, Any
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
        train_source_conf: List[Dict] = [], 
        test_source_conf: List[Dict] = [],
        preprocessing_conf: Dict = {},
        batch_size: Dict[str, int] = {'train': 0, 'val': 0, 'test': 0},
        val_split: float = 0.0
    ):
        """_summary_

        Parameters
        ----------
        train_source_conf : List[Dict], optional
            _description_, by default []
        test_source_conf : List[Dict], optional
            _description_, by default []
        preprocessing_conf : Dict, optional
            _description_, by default {}
        batch_size : Dict[str, int], optional
            _description_, by default {}
        val_split : float, optional
            _description_, by default 0.0
        """        
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
            by default {'train': 0, 'val': 0, 'test': 0}
        val_split : float, optional
            Fraction of training data serving for validation, by default 0.0
        """
        super(MultiSourceData, self).__init__()
        self.train = train_source_conf
        self.test = test_source_conf
        self.preproc_conf = preprocessing_conf
        if self.preproc_conf is not None:
            self.preproc_fn = getattr(ngt.data.preprocess, self.preproc_conf['script'])
        else:
            self.preproc_conf = {}
            self.preproc_conf['do_preprocessing'] = False
            self.preproc_fn = None
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