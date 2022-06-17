import torch_geometric.datasets as pygdst
from typing import Dict, List
from pygmi.data.sources.core import DataSource



class PyGDataSource(DataSource):

    def __init__(self, source: str, idx_select: List[int] = None, **pyg_kwargs: Dict):
        """Initializes a data source from a PyTorch Geometric dataset.
        kwargs are passed to the PyG initializer.

        Parameters
        ----------
        source : str
            Name of the PyG dataset to use.
        idx_select : List[int]
            indices of data objects to select, by default None
        """        
        self.source = getattr(pygdst, source)(**pyg_kwargs)
        super(PyGDataSource, self).__init__(indices=idx_select)
