import wandb
import torch
import torch.nn.functional as F
import ngt.utils.math.diffops as diffops
from torch import Tensor
from typing import List
from ngt.tasks import TaskBaseModule
from ngt.tasks.types import SDF
from ngt.utils.extract import extract_level_set
from ngt.utils.visual import plot_trisurf
from ngt.nn.encoder import Autodecoder


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
        epochs_for_plot: int = 200
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
        epochs_for_plot : int, optional
            _description_, by default 200
        """        
        super(SupervisedDistanceRegression, self).__init__(sdf_functional)
        self.dim = self.geometry.dim
        self.sal = sign_agnostic
        self.lr_sdf = lr_sdf
        self.scheduler_step = lr_sched_step
        self.gamma = lr_sched_gamma
        self.resolution = plot_resolution
        self.max_coord = plot_max_coord
        self.plot_step = epochs_for_plot
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


class EikonalIVPOptimization(TaskBaseModule):

    def __init__(
        self,
        sdf_functional: SDF,
        num_shapes: int = 1,
        condition_size: int = 256,
        lr_sdf: float = 5e-4,
        lr_autodec: float = 1e-3,
        lr_sched_step: int = 2000,
        lr_sched_gamma: float = 0.5,
        surf_loss_w: float = 1.0,
        eikonal_loss_w: float = 1e-2,
        norm_loss_w: float = 1.0,
        zero_penalty_w: float = 1e-1,
        zero_penalty_a: float = 1e2,
        plot_resolution: int = 100,
        plot_max_coord: float = 1.0
    ):       
        """_summary_

        Parameters
        ----------
        sdf_functional : SDF
            _description_
        autodecoder : Autodecoder, optional
            _description_, by default None
        lr_sdf : float, optional
            _description_, by default 5e-4
        lr_autodec : float, optional
            _description_, by default 1e-3
        lr_sched_step : int, optional
            _description_, by default 500
        lr_sched_gamma : float, optional
            _description_, by default 0.5
        surf_loss_w : float, optional
            _description_, by default 1.0
        eikonal_loss_w : float, optional
            _description_, by default 1e-2
        norm_loss_w : float, optional
            _description_, by default 1.0
        zero_penalty_w : float, optional
            _description_, by default 1e-1
        zero_penalty_a : float, optional
            _description_, by default 1e2
        plot_resolution : int, optional
            _description_, by default 100
        plot_max_coord : float, optional
            _description_, by default 1.0
        """        
        super(EikonalIVPOptimization, self).__init__(sdf_functional)
        self.dim = self.geometry.dim
        self.lr_sdf = lr_sdf
        self.scheduler_step = lr_sched_step
        self.gamma = lr_sched_gamma
        self.surf_loss_w = surf_loss_w
        self.eikonal_loss_w = eikonal_loss_w
        self.norm_loss_w = norm_loss_w
        self.zero_penalty_w = zero_penalty_w
        self.zero_penalty_a = zero_penalty_a
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
        indices, surf_sample, norm_sample, space_sample = batch

        condition = None
        if self.is_conditioned:
            condition = self.autodecoder(indices)

        x_surf = surf_sample.requires_grad_()
        x_space = space_sample.requires_grad_()
        
        surf_dist = self.geometry(x_surf, condition).view(*(x_surf.shape[:-1]), 1)
        space_dist = self.geometry(x_space, condition)

        surf_loss = 0.0
        if self.surf_loss_w > 0:
            surf_loss = self.surf_loss_w * surf_dist.abs().mean()

        eikonal_loss = 0.0
        if self.eikonal_loss_w > 0:
            grad = diffops.gradient(x_space, space_dist, self.dim)
            eikonal_loss = self.eikonal_loss_w * ((grad.norm(dim=-1) - 1).abs()).mean()

        norm_loss = 0.0
        if self.norm_loss_w > 0:
            grad = diffops.gradient(x_surf, surf_dist, self.dim)
            norm_loss = self.norm_loss_w * (1 - F.cosine_similarity(grad, norm_sample, dim=-1)).mean()

        zero_penalty = 0.0
        if self.zero_penalty_w > 0:
            zero_penalty = self.zero_penalty_w * torch.exp(self.zero_penalty_a * torch.abs(space_dist)).mean()

        loss = surf_loss + eikonal_loss + norm_loss + zero_penalty
        self.log("loss", loss)
        self.log("surf_loss", surf_loss)
        self.log("eikonal_loss", eikonal_loss)
        self.log("norm_loss", norm_loss)
        self.log("zero_penalty", zero_penalty)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        condition = None
        if self.is_conditioned:
            condition = self.autodecoder(torch.randint(0, self.autodecoder.N, ()))
        V, T = extract_level_set(self.geometry, self.dim, self.resolution, self.max_coord, self.device, condition=condition)
        fig = plot_trisurf(V, T)
        self.log("Reconstruction", wandb.Image(fig))

