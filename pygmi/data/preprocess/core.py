from typing import Callable, Dict, List
from tqdm import tqdm
from pygmi.data.sources.core import DataSource


def gather_fnames(out_dir: str, source_name: str, n: int) -> List[str]:
    """Creates a list of filenames for storing processed data points.

    Parameters
    ----------
    out_dir : str
        Root dir of preprocessing output
    source_name : str
        Name of data source being preprocessed
    n : int
        Size of data source being preprocessed

    Returns
    -------
    List[str]
        List of filepaths to write preprocessed data
    """    
    return [out_dir + '/{}_{}.pth'.format(source_name, i) for i in range(n)]

def process_source(data: DataSource, fnames: List[str], fn: Callable, fn_kwargs: Dict) -> None:
    """Runs a preprocessing function for each element in a data source.

    Parameters
    ----------
    data : DataSource
        Collection of data points (inputs to fn)
    fnames : List[str]
        List of output locations on resident memory
    fn : Callable
        Preprocessing function
    fn_kwargs : Dict
        Additional arguments to preprocessing function
    """    
    for i in tqdm(range(len(fnames)), desc='Preprocessing data with {}'.format(fn.__name__)):
        fn(data[i], fnames[i], **fn_kwargs)