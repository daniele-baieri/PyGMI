import pytorch_lightning as pl
from pygmi.types import ImplicitFunction



class TaskBaseModule(pl.LightningModule):

    def __init__(self, geom_repr: ImplicitFunction):
        """Instantiates a `TaskBaseModule`, an abstract `LightningModule` 
        optimizing an `ImplicitFunction`

        Parameters
        ----------
        geom_repr : ImplicitFunction
            The implicit geometry of the scene(s) to optimize
        """        
        super(TaskBaseModule, self).__init__()
        self.geometry = geom_repr

