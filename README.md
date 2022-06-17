# PyGMI

Following the recent trends in geometric deep learning, we release PyTorch Geometric Implicit (PyGMI): a toolbox to facilitate operating with neural implicit geometric representations. 

# Usage

Besides a vast utility library spanning from volume rendering to batched differential operations, PyGMI features the following modules:

* `pygmi.nn`: a torch-based collection of popularly used neural network models in the field
* `pygmi.types`: an object-oriented interface to represent implicit functions
* `pygmi.data`: a data utility interface which allows to load data from heterogenous sources and run preprocessing algorithms
* `pygmi.tasks`: a collection of out-of-the-box PyTorch Lightning Modules allowing to quickly solve popular implicit geometry tasks

We document each submodule individually.

## Data Pipeline

## Neural Networks

## Tasks

## Implicit Functions


# Dependencies

Dependencies are preinstalled using our installer. Anyway, this is an exhaustive list:

* PyTorch

* PyTorch Lightning

* PyTorch Geometric

* Only for preprocessing:

    * https://test.pypi.org/project/cgal/ (SDF)