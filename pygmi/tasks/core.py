import pytorch_lightning as pl
from pygmi.types import ImplicitFunction



class TaskBaseModule(pl.LightningModule):

    def __init__(self, geom_repr: ImplicitFunction) -> None:
        super(TaskBaseModule, self).__init__()
        self.geometry = geom_repr

