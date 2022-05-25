import os
from typing import List



def validate_fnames(paths: List[str]) -> bool:
    """
    Verifies that a list of paths exists as files in memory.

    Parameters
    ----------
    paths : str
        A list of file paths

    Returns
    -------
    bool
        True if every path leads to a file, False ow
    """    
    for f in paths:
        if not os.path.isfile(f):
            return False
    return True