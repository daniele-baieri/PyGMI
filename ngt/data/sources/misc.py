import os
from typing import List
from torch_geometric.data import Data as PyGData
from torch_geometric.io import read_ply
from torchvision.datasets import ImageFolder
from ngt.data.sources.core import DataSource


class PLYDataSource(DataSource):

    def __init__(self, source: str, idx_select: List[int] = None):
        """Initializes a data source from .ply files in a folder.
        
        Parameters
        ----------
        source : str
            path to directory containing the files
        idx_select : List[int], optional
            indices of data objects to select, by default None
        """           
        self.source = os.listdir(source)
        super(PLYDataSource, self).__init__(indices=idx_select)
    
    def process(self, obj: str) -> PyGData:
        return read_ply(obj)


class PNGDataSource(DataSource):

    def __init__(self, source: str, idx_select: List[int] = None):
        """Initializes a data source from .png files in a folder.
        
        Parameters
        ----------
        source : str
            path to directory containing the files
        idx_select : List[int], optional
            indices of data objects to select, by default None
        """           
        self.source = ImageFolder(source)
        super(PNGDataSource, self).__init__(indices=idx_select)

    