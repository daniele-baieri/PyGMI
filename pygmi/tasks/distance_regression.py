import wandb
import torch
from torch import Tensor
from typing import List
from pygmi.tasks import TaskBaseModule
from pygmi.types import SDF
from pygmi.utils.extract import extract_level_set
from pygmi.utils.visual import plot_trisurf
from pygmi.nn.encoder import Autodecoder



class SupervisedDistanceRegression(TaskBaseModule):

    def __init__(
        self,
        sdf_functional: SDF,
        num_shapes: int = 1,
        condition_size: int = 256,
        sign_agnostic: bool = True,
        lr_sdf: float = 5e-4,
        lr_autodec: float = 1e-3,
        lr_sched_step: int = 500,
        lr_sched_gamma: float = 0.5,
        latent_loss_w: float = 1e-3,
        plot_resolution: int = 100,
        plot_max_coord: float = 1.0,
    ):       
        """Instantiates a `SupervisedDistanceRegression` task. This tasks reconstructs
        SDFs from labeled point clouds by regression from a signal over points in space.

        Parameters
        ----------
        sdf_functional : SDF
            Tensor functional representing a signed distance function
        num_shapes : int, optional
            Support for multi-shape optimization, by default 1
        condition_size : int, optional
            Dimension of latent vectors for multi-shape optimization, by default 256
        sign_agnostic : bool, optional
            Whether the training data is signed (False) or unsigned (True), by default True
        lr_sdf : float, optional
            Learning rate for SDF optimization, by default 5e-4
        lr_autodec : float, optional
            Learning rate for latent vectors optimization, by default 1e-3
        lr_sched_step : int, optional
            Step LR scheduler - size of steps, by default 500
        lr_sched_gamma : float, optional
            Step LR scheduler - decay factor, by default 0.5
        latent_loss_w : float, optional
            Weight of zero-mean constraint for latent vectors, by default 1e-3
        plot_resolution : int, optional
            Grid resolution of mesh extraction for plots, by default 100
        plot_max_coord : float, optional
            Maximum absolute coordinate of plot figures, by default 1.0
        """        
        super(SupervisedDistanceRegression, self).__init__(sdf_functional)
        self.dim = self.geometry.dim
        self.sal = sign_agnostic
        self.lr_sdf = lr_sdf
        self.scheduler_step = lr_sched_step
        self.gamma = lr_sched_gamma
        self.latent_loss_w = latent_loss_w
        self.resolution = plot_resolution
        self.max_coord = plot_max_coord
        if num_shapes > 1:
            self.autodecoder = Autodecoder(num_shapes, condition_size)
            self.is_conditioned = True
            self.lr_autodec = lr_autodec

    def configure_optimizers(self) -> List:
        opt_params = [{'params': self.geometry.parameters(), 'lr': self.lr_sdf}]
        if self.is_conditioned:
            opt_params.append({'params': self.autodecoder.parameters(), 'lr': self.lr_autodec})
        optimizer = torch.optim.Adam(opt_params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.scheduler_step, self.gamma)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx) -> Tensor:
        indices, _, _, dist_sample = batch
        x_space, y_space = dist_sample[..., :self.dim], dist_sample[..., self.dim:]

        condition = None
        latent_loss = 0.0
        if self.is_conditioned:
            condition = self.autodecoder(indices)
            if self.latent_loss_w > 0:
                latent_loss = self.latent_loss_w * condition.norm(dim=-1).mean()

        sdf = self.geometry(x_space, condition)

        if self.sal:
            sdf_loss = (sdf.view_as(y_space).abs() - y_space).abs().mean()
        else:
            sdf_loss = (sdf.view_as(y_space) - y_space).abs().mean()

        loss = sdf_loss + latent_loss
        self.log("loss", loss)
        self.log("sdf_loss", sdf_loss)
        self.log("latent_loss", latent_loss)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        condition = None
        if self.is_conditioned:
            condition = self.autodecoder(torch.randint(0, self.autodecoder.N, ()))
        V, T = extract_level_set(self.geometry, self.dim, self.resolution, self.max_coord, self.device, condition=condition)
        fig = plot_trisurf(V, T)
        self.log("Reconstruction", wandb.Image(fig))