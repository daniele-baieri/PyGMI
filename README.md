# PyGMI

Following the recent trends in geometric deep learning, we release PyTorch Geometric Implicit (PyGMI): a toolbox to facilitate operating with neural implicit geometric representations. 

# Installation

Download this repository, cd to top level folder, and run:
```
    pip install --extra-index-url=https://test.pypi.org/simple/ .
```
This will install PyGMI, along with several other dependencies, to the current shell's Python directory. We advise to do this in a virtual environment.

The installation procedure requires you to install PyTorch Geometric, which cannot be automatically installed (yet).
We refer to [the library's documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), 
but we advise to run the simple install procedure with conda:
```
    conda install pyg -c pyg
```
which will auto-detect your torch and cuda versions.



# Usage

Besides a vast utility library spanning from volume rendering to batched differential operations, PyGMI features the following modules:

* `pygmi.nn`: a torch-based collection of popularly used neural network models in the field
* `pygmi.types`: an object-oriented interface to represent implicit functions
* `pygmi.data`: a data utility interface which allows to load data from heterogenous sources and run preprocessing algorithms
* `pygmi.tasks`: a collection of out-of-the-box PyTorch Lightning Modules allowing to quickly solve popular implicit geometry tasks

We document each submodule individually.

## Neural Networks

Just a collection of `torch.nn.Module`s. It features popular neural architectures used in neural implicit geometry. Most objects in this module can be initialized without constructor parameters: they will take as defaults the values shown in the original paper/implementation. SDF nets have fixed output dimension 1 and default input dimension 3. A comprehensive list:

* `DeepReLUSDFNet`: An 8-layer MLP with 512 hidden units, ReLU activation, spherical weight initialization and skip connection at layer 4. 
* `SmoothDeepSDFNet`: An 8-layer MLP with 512 hidden units, SoftPlus activation, spherical weight initialization and skip connection at layer 4. 
* `SirenMLP`: Siren (sine-activated MLP) networks base class. __This is general purpose and has no defaults__. The `w_0` constant is set to 30 as motivated in original paper.
* `SirenSDF`: A 5-layer MLP with 256 hidden units, Sine activation with phase `w_0=30`, and sine weight initialization.
* `NeRFMLP` (and others): **To be released!**

As an additional features, we include implementations of simple shape encoding architectures.

* `Autodecoder`: a collection of trainable latent vectors, one for each data point.
* `PointNet2Encoder`: a convenient implementation of PointNet++. Uses farthest point sampling for choosing pivots and radius clustering for convolution.

## Data Pipeline

The data interface is completely optional to basic PyGMI usage, but it is applied in our pre-implemented Tasks. We mainly developed it for the common need of integrating data from multiple homogenous data sources (e.g. multiple 3D shape datasets).


## Tasks

## Implicit Functions

## Utilities