import os
import warnings
import sys
import argparse
from typing import List
from setuptools import setup
from setuptools.command.install import install


if __name__ == "__main__":
    
    setup()


'''
SCATTER = 'torch-scatter'
SPARSE = 'torch-sparse'
GEOMETRIC = 'torch-geometric -f https://data.pyg.org/whl/torch-{}+{}.html"'


class InstallCommand(install):

    user_options = install.user_options + [
        ('full-feature-set', None, 'whether to install additional dependencies with pip')
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.full_feature_set = False

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        global ffs
        ffs = self.full_feature_set
        install.run(self)
        
        import torch
        try:
            import torch_geometric
        except ImportError:
            vrs = torch.__version__
            cuda = 'cu' + torch.version.cuda.replace('.', '') if torch.cuda.is_available() else 'cpu'
            deps = [SCATTER, SPARSE, GEOMETRIC.format(vrs, cuda)]

        if ffs:
            for d in deps:
                os.system('pip install {}'.format(d))
        else:
            deps = '\n\t'.join(deps)
            print('Additional dependencies:\n{}\nWill not be installed. Install them separately to enable the full PyGMI feature set.'.format(deps))



if __name__ == "__main__":

    setup(cmdclass={'install': InstallCommand})
'''



# PyGMI 
