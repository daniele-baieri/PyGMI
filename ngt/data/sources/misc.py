import os
from typing import List
from ngt.data.sources.core import DataSource


class InMemoryDataSource(DataSource):

    def __init__(self, path: str, idx_select: List[int] = None):
        """
        Initializes a data source from files.
        

        Parameters
        ----------
        path : str
            path to directory containing the files
        idx_select : List[int], optional
            indices of data objects to select, by default None
        """           
        super(InMemoryDataSource, self).__init__(idx_select)

        self.source = os.listdir(path)
    