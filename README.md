# PyGMI

Following the recent trends in geometric deep learning, we release PyTorch Geometric Implicit (PyGMI): a toolbox to facilitate operating with neural implicit geometric representations. 

# Installation

**We recommend to install in a Python environment with a working PyTorch (>= 1.8.0) installation**. If PyTorch is not found, the installer will automatically pick a minimal CPU installation. To install PyGMI, download this repository, cd to top level folder, and run:
```
pip install --extra-index-url=https://test.pypi.org/simple/ . --verbose
```
This will install PyGMI, along with several other dependencies, to the current shell's Python directory. We advise to do this in a virtual environment.

## Additional dependencies

Our setup script will run additional silent `pip install`s for PyTorch Geometric and its dependencies, if it is not found in the installation environment. If you do not wish to run these code lines, set the environment variable `PYGMI_NO_ADD_DEPS` to `1`. Then, to install PyTorch Geometric, we refer to [the library's documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), but we advise to run the simple install procedure with conda:
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

The data interface is completely optional to basic PyGMI usage, but it is applied in our pre-implemented Tasks. We mainly developed it for the common need of integrating data from multiple homogenous data sources (e.g. multiple 3D shape datasets). At the moment, our data pipeline only supports in-memory datasets (i.e. data which is preprocessed, stored on disk and loaded on-demand).

While we suggest to follow the usage showed in `examples/`, a simple `pygmi.data` use case can be described as follows:

1. Select any number of data sources, each with its type (must be a class name in `pyg.data.sources`), configuration and index selection (to cherry-pick dataset elements)
2. Define a preprocessing function or select one from `pygmi.data.preprocess`; this function should take raw data as input and save processed data to disk *
3. Subclass `pygmi.data.dataset.MultiSourceData` to override:
    1. The `collate` method, defining how a list of data points is aggregated to form a batch *
    2. The `load_data_point` method, defining how a preprocessed data point is loaded from disk to main memory *
5. Instantiate your subclass by passing the data sources (see below) and the preprocessing information (see below) as `dict`s


\* This behaviour may change in the future.


Data source configuration example:
```
### Create a dataset using both FAUST splits (train and test) as training dataset. ###

train_source_conf=[
    dict(
        type='PyGDataSource',
        source_conf=dict(
            source='FAUST',
            idx_select=None,
            root='/path/to/FAUST/dir',  # kwargs for PyG FAUST constructor 
            train=True                  # kwargs for PyG FAUST constructor 
        )
    ),
    dict(
        type='PyGDataSource',
        source_conf=dict(
            source='FAUST',
            idx_select=None,
            root='/path/to/FAUST/dir',  # kwargs for PyG FAUST constructor 
            train=False                 # kwargs for PyG FAUST constructor 
        )
    )  
]
```

Preprocessing configuration example:
```
### Require computation of 500.000 ground truth distance values for each shape ###

preprocessing_conf=dict(
    do_preprocessing=True,
    out_dir='path/to/data/output/',
    script='get_distance_values',
    conf=dict(sample=500000)
)
```
You should always supply at least `out_dir` and `do_preprocessing`; if preprocessing is not required, PyGMI will look for saved preprocessed files in `out_dir`. An error will be raised if preprocessing is disabled and there are no files to load.


## Tasks

You may regard the `tasks` submodule as a collection of algorithms to solve popular implicit geometry tasks (e.g. surface reconstruction, view synthesis). 
As for `nn`, we tried to define as many default parameter values as possible, following the values found in original publications/code releases.

In general, all tasks take as input a `pygmi.types.ImplicitFunction` object and run some optimization task on it. Once the process is completed, you may access
`task.geometry` to recover the optimized implicit function along with its interface methods. 

Since tasks inherit from PyTorch Lightning Modules, you may execute them by creating a Trainer and a LightningDataModule (`pygmi.data.dataset` objects comply with this standard)
and calling `trainer.fit(task, data)`. Take a look at `examples/` for a more concrete demonstration!


## Implicit Functions

`pygmi.types` defines abstract object-oriented interfaces to work with `ImplicitFunction` subclasses. The real computation occurs in the `approximator` object which is required at creation. This could be a neural network or an analytic function, such as `pygmi.utils.sphere_sdf`. 



## Utilities

Our `utils` submodule is structured as follows:

* `utils`
    * `files`: filesystem operations (useful for, e.g., preprocessing functions)
    * `misc`: various generic functions with no precise application
    * `extract`: extraction of explicit geometry from implicit representations
        * `core`: functions that can be applied to most implicit representations (generic level set operations)
    * `math`: predefined math operations
        * `diffops`: batched differential operations such as gradient, Jacobian, Hessian, etc. All are computed using `autograd`, optionally allowing higher-order differentiation
    * `visual`: visualization utilities
        * `core`: generic figure handling functions, common 3D plots use-cases (isosurfaces)

All functions can be accessed directly from `pygmi.utils`, without further nesting. 
