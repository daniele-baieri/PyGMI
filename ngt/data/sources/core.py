from typing import List, Any


class DataSource:

    def __init__(self, indices: List[int] = None):
        """Initialize abstract data source. 
        If no indices are selected, use all the available data.

        Parameters
        ----------
        indices : List[int], optional
            indices of data objects to select, by default None
        """           
        self.indices = range(len(self.source)) if indices is None else indices

    def __getitem__(self, idx: int) -> Any:
        return self.source[self.indices[idx]]

    def __len__(self) -> int:
        return len(self.indices)
    
    def process(self, obj: Any) -> Any:
        return obj
