import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pygmi.data.dataset import SDFSupervisedData
from pygmi.tasks import SupervisedDistanceRegression
from pygmi.types import SDF
from pygmi.nn import DeepReLUSDFNet
from pygmi.utils.extract import grid_evaluation
from pygmi.utils.visual import isosurf_animation


"""Learns a parametric SDF for the FAUST train + test dataset.
Preprocesses the data from its PyG counterpart, then runs
sign agnostic distance regression. Finally, the result for one
of the training shapes is plotted.
"""

if __name__ == "__main__":

    logging = False
    gpu = 1 if torch.cuda.is_available() else 0
    
    data = SDFSupervisedData(
        train_source_conf=[
            dict(
                type='PyGDataSource',
                source_conf=dict(
                    source='FAUST',
                    idx_select=None,
                    pyg_kwargs=dict(
                        root='path/to/FAUST/dir',
                        train=True
                    )
                )
            ),
            dict(
                type='PyGDataSource',
                source_conf=dict(
                    source='FAUST',
                    idx_select=None,
                    pyg_kwargs=dict(
                        root='path/to/FAUST/dir',
                        train=False
                    )
                )
            )  
        ],
        preprocessing_conf=dict(
            do_preprocessing=True,
            out_dir='path/to/data/output/',
            script='get_distance_values',
            conf=dict(sample=500000)
        ),
        batch_size=dict(train=16, val=1, test=1),
        use_normals=False
    )

    num_shapes = len(data)
    latent_dim = 256
    net = DeepReLUSDFNet(input_dim = 3 + latent_dim)
    sdf = SDF(net)
    task = SupervisedDistanceRegression(sdf, num_shapes=num_shapes, condition_size=latent_dim)

    epochs = 2000
    if logging is True:
        logger = WandbLogger(project='PyGMI Task Logs')
    else: 
        logger = False
    trainer = pl.Trainer(logger=logger, max_epochs=epochs, gpus=gpu)
    trainer.fit(task, data)

    shape_to_plot = 16
    latent = task.autodecoder(shape_to_plot)
    volume = grid_evaluation(sdf, 3, 100, 1.2, 'cuda' if gpu == 1 else 'cpu', condition=latent)
    fig = isosurf_animation(volume, axes=[-1.2, 1.2] * 3, steps=10, min_level=-0.5, max_level=0.7)
    fig.show()