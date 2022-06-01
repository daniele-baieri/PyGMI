import pytorch_lightning as pl
from typing import Union, Optional, Callable, Any
from ngt.tasks.types import ImplicitFunction



class TaskBaseModule(pl.LightningModule):

    def __init__(self, geom_repr: ImplicitFunction, debug_mode: bool = False) -> None:
        super(TaskBaseModule, self).__init__()
        self.geometry = geom_repr
        self.debug = debug_mode

    def log(
        self, 
        name: str, 
        value: Any, 
        prog_bar: bool = False, 
        logger: bool = True, 
        on_step: Optional[bool] = None, 
        on_epoch: Optional[bool] = None, 
        reduce_fx: Union[str, Callable] = "default", 
        tbptt_reduce_fx: Optional[Any] = None, 
        tbptt_pad_token: Optional[Any] = None, 
        enable_graph: bool = False, 
        sync_dist: bool = False, 
        sync_dist_op: Optional[Any] = None, 
        sync_dist_group: Optional[Any] = None, 
        add_dataloader_idx: bool = True, 
        batch_size: Optional[int] = None, 
        metric_attribute: Optional[str] = None, 
        rank_zero_only: Optional[bool] = None
    ) -> None:
        if self.debug:
            return super(TaskBaseModule, self).log(
                name, value, prog_bar, logger, on_step, on_epoch, 
                reduce_fx, tbptt_reduce_fx, tbptt_pad_token, enable_graph, 
                sync_dist, sync_dist_op, sync_dist_group, add_dataloader_idx, 
                batch_size, metric_attribute, rank_zero_only)