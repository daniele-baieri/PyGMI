import torch
from torch import Tensor
from torch.autograd import grad


def gradient(inputs: Tensor, outputs: Tensor, dim: int = 3) -> Tensor:
    """Computes the gradient of `outputs` wrt `inputs`. `inputs` must require grad
    and there must be a path from `outputs` to `inputs` in the computational graph.

    Parameters
    ----------
    inputs : Tensor
        List of input vectors. Shape `B_1 x ... x B_n x dim`
    outputs : Tensor
        List of output scalar values. Shape `B_1 x ... x B_n x 1`
    dim : int, optional
        Dimensionality of input vectors, by default 3

    Returns
    -------
    Tensor
        Point-wise gradient
    """  
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)
    return points_grad[0][..., :dim]

def divergence(inputs: Tensor, outputs: Tensor, dim: int = 3) -> Tensor:
    """Computes the divergence of `outputs` wrt `inputs`. `inputs` must require grad
    and there must be a path from `outputs` to `inputs` in the computational graph.

    Parameters
    ----------
    inputs : Tensor
        List of input vectors. Shape `B_1 x ... x B_n x dim`
    outputs : Tensor
        List of output vectors. Shape `B_1 x ... x B_n x dim`
    dim : int, optional
        Dimensionality of input and output vectors, by default 3

    Returns
    -------
    Tensor
        Point-wise divergence
    """    

    gradients = torch.zeros_like(outputs, dtype=torch.float, device=outputs.device)
    d_points = torch.ones((*outputs.shape[:-1], 1), requires_grad=False, device=outputs.device)
    for d in range(dim):
        gradients[..., d] = grad(
            outputs=outputs[..., d:d+1], 
            inputs=inputs, 
            grad_outputs=d_points, 
            retain_graph=True, 
            create_graph=True,
            only_inputs=True)[0][..., d]
    return gradients.sum(dim=-1, keepdim=True)

def jacobian(inputs: Tensor, outputs: Tensor) -> Tensor:
    """Computes the Jacobian of `outputs` wrt `inputs`. `inputs` must require grad
    and there must be a path from `outputs` to `inputs` in the computational graph.

    Parameters
    ----------
    inputs : Tensor
        List of input vectors. Shape `B_1 x ... x B_n x dim`
    outputs : Tensor
        List of output vectors. Shape `B_1 x ... x B_n x dim`
        
    Returns
    -------
    Tensor
        Point-wise Jacobian
    """  
    in_d = inputs.shape[-1]
    out_d = outputs.shape[-1]
    J = torch.zeros((*outputs.shape[:-1], in_d, out_d), dtype=torch.float, device=outputs.device)
    d_points = torch.ones((*outputs.shape[:-1], 1), requires_grad=False, device=outputs.device)
    for d in range(out_d):
        J[..., :, d] = grad(
            outputs=outputs[..., d:d+1], 
            inputs=inputs, 
            grad_outputs=d_points, 
            retain_graph=True, 
            create_graph=True,
            only_inputs=True)[0]
    return J

def hessian(inputs: Tensor, outputs: Tensor, dim: int = 3, diff: bool = True) -> Tensor:
    """Computes the Hessian of `outputs` wrt `inputs` as the Jacobian of the gradient.
    `inputs` must require grad and there must be a path from `outputs` to `inputs` 
    in the computational graph.

    Parameters
    ----------
    inputs : Tensor
        List of input vectors. Shape `B_1 x ... x B_n x dim`
    outputs : Tensor
        List of output scalar values. Shape `B_1 x ... x B_n x 1`
    dim : int, optional
        Dimensionality of input points, by default 3
    diff: bool, optional
        Whether the return value should be differentiable, by default True

    Returns
    -------
    Tensor
        Point-wise Hessian
    """  
    H = torch.zeros((*outputs.shape[:-1], dim, dim), dtype=torch.float, device=outputs.device)
    G = gradient(inputs, outputs, dim=dim).view_as(inputs)  # .squeeze()
    d_points = torch.ones_like(outputs, requires_grad=False)
    for i in range(dim):  # Gradient of gradient in each dimension
        H[..., i, :] = torch.autograd.grad(
            outputs=G[..., i:i+1],
            inputs=inputs,
            create_graph=diff,
            retain_graph=True,
            grad_outputs=d_points,
            only_inputs=True
        )[0]
    return H
