import yaml
import pytorch_lightning as pl
import ngt.data.dataset
import ngt.tasks
import ngt.tasks.types
import ngt.nn
from typing import Tuple
from pytorch_lightning.loggers import WandbLogger
from ngt.tasks.types import ImplicitFunction



class TaskBaseModule(pl.LightningModule):

    def __init__(self, geom_repr: ImplicitFunction) -> None:
        super(TaskBaseModule, self).__init__()
        self.geometry = geom_repr


def run_task(
    data: pl.LightningDataModule, task_module: TaskBaseModule, trainer: pl.Trainer
) -> ImplicitFunction:
    """_summary_

    Parameters
    ----------
    data : pl.LightningDataModule
        _description_
    task_module : TaskBaseModule
        _description_
    trainer : pl.Trainer
        _description_

    Returns
    -------
    ImplicitFunction
        _description_
    """    
    trainer.fit(task_module, data)
    return task_module.geometry

def make_environment(conf_file: str) -> Tuple[pl.LightningDataModule, TaskBaseModule, pl.Trainer]:
    conf = yaml.safe_load(open(conf_file))

    # Make data
    data_conf = conf['data']
    make_data = getattr(ngt.data.dataset, data_conf['dataset_type'])
    del data_conf['dataset_type']
    data = make_data(**data_conf)

    # Make task
    task_conf = conf['task']
    make_task = getattr(ngt.tasks, task_conf['name'])
    del task_conf['name']

    func_conf = task_conf['functional']
    make_functional = getattr(ngt.tasks.types, func_conf['type'])
    del func_conf['type']

    nn_conf = func_conf['approximator']
    make_nn = getattr(ngt.nn, nn_conf['type'])
    del nn_conf['type']

    nn = make_nn(**nn_conf)
    func = make_functional(nn, **func_conf)
    task = make_task(func, **task_conf)

    # Make trainer
    opt_conf = conf['optimization']
    if opt_conf['logging'] is True:
        logger = WandbLogger(project='NGT Task Logs', config=conf)
    else:
        logger = False
    del opt_conf['logging']
    opt_conf['logger'] = logger
    trainer = pl.Trainer(**opt_conf)

    return data, task, trainer