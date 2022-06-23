import torch
import torch.nn as nn
import numpy as np
from torch import Tensor


### Adapted from official Siren implementation https://github.com/vsitzmann/siren ###

def first_layer_sine_init(m: nn.Module) -> None:
    """Special first layer initialization for Sine-activate MLPs.

    Parameters
    ----------
    m : nn.Module
        Linear layer to initialize
    """    
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.uniform_(m.weight, -1 / num_input, 1 / num_input)


def sine_init(m: nn.Module, w0: float = None) -> None:
    """Special initialization for layers of Sine-activated MLPs.

    Parameters
    ----------
    m : nn.Module
        Linear layer to initialize
    w0 : float, optional
        Phase factor which can be leveraged in initialization, by default None
    """    
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            if w0 is None:
                nn.init.uniform_(m.weight, -np.sqrt(6 / num_input), np.sqrt(6 / num_input))
            else:
                nn.init.uniform_(m.weight, -np.sqrt(6 / (num_input * (w0 ** 2))), np.sqrt(6 / (num_input * (w0 ** 2))))

        
class Sine(nn.Module):

    def __init__(self, w0: float):
        """Initializes a Sine activation function with phase factor `w0`

        Parameters
        ----------
        w0 : float
            Phase factor
        """        
        super(Sine, self).__init__()
        self.w = w0

    def forward(self, input: Tensor) -> Tensor:
        """Runs Sine activation function over `input`

        Parameters
        ----------
        input : Tensor
            A (partially processed) batch of data points

        Returns
        -------
        Tensor
            Point-wise sine of the scalar product of `self.w` (phase factor) and `input`
        """        
        return torch.sin(self.w * input)


class SirenMLP(nn.Module):

    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        hidden_dim: int, 
        hidden_layers: int,
        w0: float,
        init_w0: float,
        use_first_layer_init: bool,
        w0_in_layer_init: bool
    ):
        """Creates as fully-configurable Siren MLP as presented in https://arxiv.org/abs/2006.09661.


        Parameters
        ----------
        in_dim : int
            Number of input features
        out_dim : int
            Number of output features
        hidden_dim : int
            Hidden dimension, same for all layers
        hidden_layers : int
            Number of hidden layers
        w0 : float
            Sine phase amplification factor (layers 2+)
        init_w0 : float
            Sine phase amplification factor (layer 1)
        use_first_layer_init : bool
            Use a special initialization for first layer weights
        w0_in_layer_init : bool
            Leverage w0 in layer initialization, as proposed in Siren supmat
        """        
        super(SirenMLP, self).__init__()

        self.init_w0, self.w0 = init_w0, w0
        self.init_actvn, self.actvn = Sine(self.init_w0), Sine(self.w0)

        self.net = nn.ModuleList()
        self.net.append(nn.Linear(in_dim, hidden_dim))

        if use_first_layer_init:
            first_layer_sine_init(self.net[-1])
        else:
            w = self.init_w0 if w0_in_layer_init else None
            sine_init(self.net[-1], w)
        
        w = self.w0 if w0_in_layer_init else None
        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_dim, hidden_dim))
            sine_init(self.net[-1], w)
        self.net.append(nn.Linear(hidden_dim, out_dim))
        sine_init(self.net[-1], w)

    def forward(self, x_in: Tensor) -> Tensor:
        """Runs the model over a sample of points.

        Parameters
        ----------
        x_in : Tensor
            Sample of points, shape `B_1 x ... x B_n x I`

        Returns
        -------
        Tensor
            Output for each point, shape `B_1 x ... x B_n x O`
        """        
        h = self.init_actvn(self.net[0](x_in))
        for layer in self.net[1:-1]:
            h = self.actvn(layer(h))
        h = self.net[-1](h)
        return h


class SirenSDF(SirenMLP):

    def __init__(
        self, 
        in_dim: int = 3, 
        hidden_dim: int = 256, 
        hidden_layers: int = 5,
        use_first_layer_init: bool = False,
        w0_in_layer_init: bool = True
    ):
        """Preconfigured Siren MLP for SDF tasks. w0 is fixed to 30 as motivated
        in original paper. Parameter defaults are as in original implementation.

        Parameters
        ----------
        in_dim : int, optional
            Number of input features, by default 3
        hidden_dim : int, optional
            Hidden dimension, same for all layers, by default 256
        hidden_layers : int, optional
            Number of hidden layers, by default 5
        use_first_layer_init : bool, optional
            Use a special initialization for first layer weights, by default False
        w0_in_layer_init : bool, optional
            Leverage w0 in layer initialization, by default True
        """        
        super(SirenSDF, self).__init__(
            in_dim, 1, hidden_dim, hidden_layers, 30.0, 30.0, use_first_layer_init, w0_in_layer_init
        )