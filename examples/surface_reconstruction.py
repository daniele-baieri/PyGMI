import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pygmi.data.dataset import SDFUnsupervisedData
from pygmi.tasks import EikonalIVPOptimization
from pygmi.types import SDF
from pygmi.nn import SirenSDF
from pygmi.utils.extract import grid_evaluation
from pygmi.utils.visual import isosurf_animation


"""Learns a SDF from a single .txt point cloud, using a Siren
network and unsupervised training. The result is plotted after
optimization.
"""

if __name__ == "__main__":

    logging = False
    gpu = 1 if torch.cuda.is_available() else 0
    
    has_normal_data = True
    data = SDFUnsupervisedData(
        train_source_conf=[
            dict(
                type='TXTArrayDataSource',
                source_conf=dict(
                    source='/path/to/dir/containing/txt/file',
                    idx_select=None
                )
            )
        ],
        preprocessing_conf=dict(
            do_preprocessing=True,
            out_dir='/path/to/data/output/', 
            script='center_point_cloud',
            conf=dict(mnfld_sigma=True)
        ),
        batch_size=dict(train=1, val=1, test=1),
        use_normals=has_normal_data,
        surf_sample=30000,
        global_space_sample=3750
    )

    net = SirenSDF()
    sdf = SDF(net)
    task = EikonalIVPOptimization(sdf, lr_sdf=1e-4, lr_sched_step=None, lr_sched_gamma=None)

    epochs = 5000
    if logging is True:
        logger = WandbLogger(project='PyGMI Task Logs')
    else: 
        logger = False
    trainer = pl.Trainer(logger=logger, max_epochs=epochs, accelerator='gpu' if gpu == 1 else 'cpu', devices=gpu)
    trainer.fit(task, data)

    net = net.to('cuda' if gpu == 1 else 'cpu')
    volume = grid_evaluation(sdf, 3, 100, 1.2, 'cuda' if gpu == 1 else 'cpu')
    fig = isosurf_animation(volume, axes=[-1.2, 1.2] * 3, steps=10, min_level=-0.5, max_level=0.7)
    fig.show()