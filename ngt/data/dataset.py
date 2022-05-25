import re
import random
import numpy as np
import pytorch_lightning as pl
import ngt.data.sources
import ngt.data.preprocess
from torch.utils.data import DataLoader
from torch import Tensor
from typing import Optional, List, Dict, Tuple
from ngt.data.preprocess import gather_fnames, process_source
from ngt.utils.files import validate_fnames



class MultiSourceData(pl.LightningDataModule):

    def __init__(
        self, 
        train_source_conf: List[Dict], 
        test_source_conf: List[Dict],
        preprocessing_conf: Dict,
        val_split: float
    ):
        """
        _summary_

        Parameters
        ----------
        train_source_conf : List[Dict]
            _description_
        test_source_conf : List[Dict]
            _description_
        preprocessing_conf : Dict
            _description_
        val_split : float
            _description_
        """    
        super(MultiSourceData, self).__init__()
        self.train = train_source_conf
        self.test = test_source_conf
        self.preproc_conf = preprocessing_conf
        self.preproc_fn = getattr(ngt.data.preprocess, self.preproc_conf['script'])
        self.train_paths, self.val_paths, self.test_paths = [], [], []
        self.val_split = val_split

    def setup(self, stage: Optional[str] = None) -> None:
        
        if stage == 'fit' or stage is None:
            for conf in self.train:
                source = getattr(ngt.data.sources, conf['type'])(**conf['source_conf'])
                sname = '_'.join(re.split('/|\\', conf['source_conf']['source'].replace('/', '_')))
                fnames = gather_fnames(self.preproc_conf['out_dir'], sname, len(source))
                if self.preproc_conf['do_preprocessing']:
                    process_source(source, self.preproc_fn, fnames, self.preproc_conf['conf'])
                else:
                    if not validate_fnames(fnames):
                        raise RuntimeError('Processed files missing with preprocessing disabled.')
                self.train_paths += fnames

            n_val = int(len(self.train_paths) * self.val_split)
            val_idxs = random.sample(range(len(self.train_paths)), n_val)
            self.val_paths = [self.train_paths[i] for i in val_idxs]
            self.train_paths = np.delete(self.train_paths, val_idxs)
                
        if stage == 'test' or stage is None:  
            for conf in self.test:
                source = getattr(ngt.data.sources, conf['type'])(**conf['source_conf'])
                sname = '_'.join(re.split('/|\\', conf['source_conf']['source'].replace('/', '_')))
                fnames = gather_fnames(self.preproc_conf['out_dir'], sname, len(source))
                if self.preproc_conf['do_preprocessing']:
                    process_source(source, self.preproc_fn, fnames, self.preproc_conf['conf'])
                else:
                    if not validate_fnames(fnames):
                        raise RuntimeError('Processed files missing with preprocessing disabled.')
                self.test_paths += fnames
        
    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError()

    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError()

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError()


class SDFUnsupervisedData(MultiSourceData):

    def __init__(
        self, 
        train_source_conf: List[Dict], 
        test_source_conf: List[Dict],
        preprocessing_conf: Dict,
        val_split: float,
        global_sigma: float = None
    ):
        """
        _summary_

        Parameters
        ----------
        train_source_conf : List[Dict]
            _description_
        test_source_conf : List[Dict]
            _description_
        preprocessing_conf : Dict
            _description_
        val_split : float
            _description_
        """    
        super(SDFUnsupervisedData, self).__init__(
            train_source_conf, test_source_conf, preprocessing_conf, val_split)
        self.preproc_fn = ngt.data.preprocess.upsample_with_normals

    def collate(self, paths: List[str]) -> Tuple[Tensor, Tensor, Tensor]:
        pass

    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError()

    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError()

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError()