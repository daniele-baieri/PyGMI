import re
import os
import random
import numpy as np
import pytorch_lightning as pl
import ngt.data.sources
import ngt.data.preprocess
from torch.utils.data import DataLoader
from typing import Callable, Optional, List, Dict
from ngt.data.sources.core import DataSource



def process_source(data: DataSource, fnames: List[str], fn: Callable, fn_kwargs: Dict) -> None:
    for i in range(len(fnames)):
        fn(data[i], **fn_kwargs)

def gather_fnames(out_dir: str, source_name: str, n: int) -> List[str]:
    return [out_dir + '/{}_{}.pth'.format(source_name, i) for i in range(n)]

def validate_fnames(dir: str) -> bool:
    files = os.listdir(dir)
    for f in files:
        if not os.path.isfile(f):
            return False
    return True


class MultiSourceData(pl.LightningDataModule):

    def __init__(
        self, 
        train_source_conf: List[Dict], 
        test_source_conf: List[Dict],
        preprocessing_conf: Dict,
        val_split: float
    ):
        super(MultiSourceData, self).__init__()
        self.train = train_source_conf
        self.test = test_source_conf
        self.preproc_conf = preprocessing_conf
        self.train_paths, self.val_paths, self.test_paths = [], [], []
        self.val_split = val_split

    def setup(self, stage: Optional[str] = None) -> None:
        
        if stage == 'fit' or stage is None:
            for conf in self.train:
                source = getattr(ngt.data.sources, conf['type'])(**conf['source_conf'])
                sname = '_'.join(re.split('/|\\', conf['source_conf']['source'].replace('/', '_')))
                fnames = gather_fnames(self.preproc_conf['out_dir'], sname, len(source))
                if self.preproc_conf['do_preprocessing']:
                    preproc_fn = getattr(ngt.data.preprocess, self.preproc_conf['script'])
                    process_source(source, preproc_fn, fnames)
                else:
                    if not validate_fnames(self.preproc_conf['out_dir']):
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
                    preproc_fn = getattr(ngt.data.preprocess, self.preproc_conf['script'])
                    process_source(source, preproc_fn, fnames)
                else:
                    if not validate_fnames(self.preproc_conf['out_dir']):
                        raise RuntimeError('Processed files missing with preprocessing disabled.')
                self.test_paths += fnames
        
    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError()

    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError()

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError()
