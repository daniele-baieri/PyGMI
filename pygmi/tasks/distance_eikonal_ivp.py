import wandb
import torch
import torch.nn.functional as F
import pygmi.utils.math.diffops as diffops
from torch import Tensor
from typing import List
from pygmi.tasks import TaskBaseModule
from pygmi.types import SDF
from pygmi.utils.extract import extract_level_set
from pygmi.utils.visual import plot_trisurf
from pygmi.nn.encoder import Autodecoder



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
        surf_loss_w: float = 3e3,
        eikonal_loss_w: float = 5e1,
        norm_loss_w: float = 1e2,
        zero_penalty_w: float = 1e2,
        zero_penalty_a: float = -1e2,
        latent_loss_w: float = 1e-3,
        plot_resolution: int = 100,
        plot_max_coord: float = 1.0
    ):       
        """Instantiates an `EikonalIVPOptimization` task. This task reconstructs geometry 
        from a point cloud by requiring the SDF to vanish on zero-level set points and to
        have unitary norm of gradient (the SDF needs to support 2nd order derivatives).
        Optionally, normal constraint (gradient at surface points equals normals) and 
        zero-value penalty (no small function values far away from surface) can be optimized for.

        Parameters
        ----------
        sdf_functional : SDF
            Tensor functional representing a signed distance function
        num_shapes : int, optional
            Support for multi-shape optimization, by default 1
        condition_size : int, optional
            Dimension of latent vectors for multi-shape optimization, by default 256
        lr_sdf : float, optional
            Learning rate for SDF optimization, by default 5e-4
        lr_autodec : float, optional
            Learning rate for latent vectors optimization, by default 1e-3
        lr_sched_step : int, optional
            Step LR scheduler - size of steps, by default 2000
        lr_sched_gamma : float, optional
            Step LR scheduler - decay factor, by default 0.5
        surf_loss_w : float, optional
            Weight of zero level set loss, by default 1.0
        eikonal_loss_w : float, optional
            Weight of eikonal loss, by default 1e-2
        norm_loss_w : float, optional
            Weight of normal loss, by default 1.0
        zero_penalty_w : float, optional
            Weight of zero value penalty, by default 1e-1
        zero_penalty_a : float, optional
            Alpha of zero value penalty, by default 1e2
        latent_loss_w : float, optional
            Weight of zero-mean constraint for latent vectors, by default 1e-3
        plot_resolution : int, optional
            Grid resolution of mesh extraction for plots, by default 100
        plot_max_coord : float, optional
            Maximum absolute coordinate of plot figures, by default 1.0
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
        self.latent_loss_w = latent_loss_w
        self.resolution = plot_resolution
        self.max_coord = plot_max_coord
        self.is_conditioned = False
        if num_shapes > 1:
            self.autodecoder = Autodecoder(num_shapes, condition_size)
            self.is_conditioned = True
            self.lr_autodec = lr_autodec

    def configure_optimizers(self) -> List:
        opt_params = [{'params': self.geometry.parameters(), 'lr': self.lr_sdf}]
        if self.is_conditioned:
            opt_params.append({'params': self.autodecoder.parameters(), 'lr': self.lr_autodec})
        optimizer = torch.optim.Adam(opt_params)
        if self.scheduler_step is None or self.gamma is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.scheduler_step, self.gamma)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx) -> Tensor:
        indices, surf_sample, norm_sample, space_sample = batch

        condition = None
        latent_loss = 0.0
        if self.is_conditioned:
            condition = self.autodecoder(indices)
            if self.latent_loss_w > 0:
                latent_loss = self.latent_loss_w * condition.norm(dim=-1).mean()

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

        loss = surf_loss + eikonal_loss + norm_loss + zero_penalty + latent_loss
        self.log("loss", loss)
        self.log("surf_loss", surf_loss)
        self.log("eikonal_loss", eikonal_loss)
        self.log("norm_loss", norm_loss)
        self.log("zero_penalty", zero_penalty)
        self.log("latent_loss", latent_loss)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        condition = None
        if self.is_conditioned:
            condition = self.autodecoder(torch.randint(0, self.autodecoder.N, ()))
        V, T = extract_level_set(self.geometry, self.dim, self.resolution, self.max_coord, self.device, condition=condition)
        fig = plot_trisurf(V, T)
        self.log("Reconstruction", wandb.Image(fig))