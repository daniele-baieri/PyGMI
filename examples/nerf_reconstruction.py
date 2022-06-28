import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pygmi.nn import NeRF

"""Learns a Neural Radiance Field from a set of images and camera poses, 
using a NeRF network and supervised training. The result is plotted after
optimization.
"""

if __name__ == "__main__":

    logging = False
    gpu = 1 if torch.cuda.is_available() else 0

    data = None

    net = NeRF()
    task = None

    epochs = 5000
    if logging is True:
        logger = WandbLogger(project='PyGMI Task Logs')
    else: 
        logger = False
    trainer = pl.Trainer(logger=logger, max_epochs=epochs, accelerator='gpu' if gpu == 1 else 'cpu', devices=gpu)
    trainer.fit(task, data)

    net = net.to('cuda' if gpu == 1 else 'cpu')
    # volume = grid_evaluation(sdf, 3, 100, 1.2, 'cuda' if gpu == 1 else 'cpu')
    # fig = isosurf_animation(volume, axes=[-1.2, 1.2] * 3, steps=10, min_level=-0.5, max_level=0.7)
    # fig.show()