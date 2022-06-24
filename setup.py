import os
import warnings
import subprocess
from setuptools import setup


NO_ADD_DEPS_KEY = 'PYGMI_NO_ADD_DEPS'
DEPS = 'torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-{}+{}.html"'


if __name__ == "__main__":

    setup()

    no_additional = os.environ[NO_ADD_DEPS_KEY] if NO_ADD_DEPS_KEY in os.environ.keys() else None


    if not no_additional or no_additional is None:
        try:
            import torch
            try:
                import torch_geometric
                warnings.warn('PyG already found, skipping.')
            except ImportError:
                vrs = torch.__version__
                cuda = 'cu' + torch.version.cuda.replace('.', '') if torch.cuda.is_available() else 'cpu'
                deps = DEPS.format(vrs, cuda)
                subprocess.call(['pip', 'install'] + deps.split(' ')) 
        except:
            warnings.warn('PyTorch not available. PyTorch Geometric will not be installed.')

    else:
        warnings.warn('Additional dependencies install is disabled. Exiting.')

# PyGMI 
