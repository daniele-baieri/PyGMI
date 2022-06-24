import os
from typing import List
from torch_geometric.data import Data as PyGData
from torch_geometric.io import read_txt_array, read_ply
from torchvision.datasets import ImageFolder
from pygmi.data.sources.core import DataSource


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
        self.source = [source + '/' + fp for fp in os.listdir(source)]
        super(PLYDataSource, self).__init__(indices=idx_select)
    
    def process(self, obj: str) -> PyGData:
        """Override of `process` method. Calls `torch_geometric.io.read_ply`
        to load a filepath pointing to a .ply file into a `torch_geometric.data.Data`
        object.

        Parameters
        ----------
        obj : str
            Path to a .ply file on disk.

        Returns
        -------
        PyGData
            A `torch_geometric.data.Data` object, representing a mesh or a point cloud.
            Usual attributes are `pos`, `face`, and `normal`.
        """        
        return read_ply(obj)


class TXTArrayDataSource(DataSource):

    def __init__(self, source: str, idx_select: List[int] = None):
        """Initializes a data source from .txt files in a folder,
        containing array data. Usually convenient for point clouds, 
        following the format:
        x y z u v w
        where each row contains point coordinates and normals.
        
        Parameters
        ----------
        source : str
            path to directory containing the files
        idx_select : List[int], optional
            indices of data objects to select, by default None
        """           
        self.source = [source + '/' + fp for fp in os.listdir(source)]
        super(TXTArrayDataSource, self).__init__(indices=idx_select)
    
    def process(self, obj: str) -> PyGData:
        """Overrides the `process` method, by reading a .txt array and
        returning it as a `torch_geometric.data.Data` object containing
        point coordinates and (optionally) normals of a point cloud.

        Parameters
        ----------
        obj : str
            Path to a .txt file containing a point cloud saved as array, 
            with format:
                x y z [u v w]
            Where (x, y, z) are the point coordinates and (u, v, w) are the
            (optional) surface normals of the corresponding points.

        Returns
        -------
        PyGData
            A `torch_geometric.data.Data` with attributes `pos` and `normal`.
        """        
        t = read_txt_array(obj)
        pos = t[:, :3]
        if t.shape[1] > 3:
            normal = t[:, 3:]
        return PyGData(pos=pos, normal=normal)


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

    