import torch_geometric.datasets as pygdst
from torch_geometric.data import Data as PyGData
from typing import Dict, List



class PyGDataSource:

    def __init__(self, source: str, idx_select: List[int] = None, **kwargs: Dict):
        """
        Initializes a data source from a PyTorch Geometric dataset.
        kwargs are passed to the PyG initializer.

        Parameters
        ----------
        source : str
            Name of the PyG dataset to use.
        idx_select : List[int]
            indices of data objects to select, by default None
        """        
        super(PyGDataSource, self).__init__()

        self.source = getattr(pygdst, source)(**kwargs)
        self.indices = idx_select