# Neural Geometry Toolkit

Following the recent trends in geometric deep learning, we release a toolbox to facilitate operating with neural implicit geometric representations. 

# Usage

We define both pre-made commonly used structures and supertypes to define your custom behaviours. 

## Configuration

## Data Pipeline


* `MultiSourceData`: a PyTorch Lightning DataModule, which you can subclass to define `{train, val, test}_dataloader` methods. Your subclasses can still use the same configurations as before.

## Neural networks

## Tasks


# Dependencies

* PyTorch

* PyTorch Lightning

* PyTorch Geometric

* Only for preprocessing:

    * https://test.pypi.org/project/cgal/ (SDF)