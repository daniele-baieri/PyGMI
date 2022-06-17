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
        plot_resolution: int = 100,
        plot_max_coord: float = 1.0,
    ):       
        """_summary_

        Parameters
        ----------
        sdf_functional : SDF
            _description_
        num_shapes : int, optional
            _description_, by default 1
        condition_size : int, optional
            _description_, by default 256
        sign_agnostic : bool, optional
            _description_, by default True
        lr_sdf : float, optional
            _description_, by default 5e-4
        lr_autodec : float, optional
            _description_, by default 1e-3
        lr_sched_step : int, optional
            _description_, by default 500
        lr_sched_gamma : float, optional
            _description_, by default 0.5
        plot_resolution : int, optional
            _description_, by default 100
        plot_max_coord : float, optional
            _description_, by default 1.0
        """        
        super(SupervisedDistanceRegression, self).__init__(sdf_functional)
        self.dim = self.geometry.dim
        self.sal = sign_agnostic
        self.lr_sdf = lr_sdf
        self.scheduler_step = lr_sched_step
        self.gamma = lr_sched_gamma
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
        if self.is_conditioned:
            condition = self.autodecoder(indices)
        sdf = self.geometry(x_space, condition)

        if self.sal:
            sdf_loss = (sdf.view_as(y_space).abs() - y_space).abs().mean()
        else:
            sdf_loss = (sdf.view_as(y_space) - y_space).abs().mean()

        self.log("loss", sdf_loss)
        return sdf_loss

    def validation_step(self, batch, batch_idx) -> None:
        condition = None
        if self.is_conditioned:
            condition = self.autodecoder(torch.randint(0, self.autodecoder.N, ()))
        V, T = extract_level_set(self.geometry, self.dim, self.resolution, self.max_coord, self.device, condition=condition)
        fig = plot_trisurf(V, T)
        self.log("Reconstruction", wandb.Image(fig))