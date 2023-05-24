import torch
from . import MetaModule
from . import MesaModule
from . import BaseModule

class MtMs(torch.nn.Module):
    def __init__(self, base_model, num_tasks, mesa_parameter_size=1, meta_bias=True, meta_dropout_p=0, mesa_parameters_init=None, meta_weight_init=None, meta_connections=None):
        super(MtMs, self).__init__()
        self.base_module = BaseModule(base_model)
        self.meta_module = MetaModule(self.base_module, mesa_parameter_size, meta_bias, meta_dropout_p, meta_weight_init, meta_connections)
        self.mesa_module = MesaModule(num_tasks, mesa_parameter_size, mesa_parameters_init)

    def forward(self, x, tasks):
        unique_tasks = torch.unique(tasks)
        y_parts = []
        for task_id in unique_tasks:
            task_rows = (tasks == task_id).nonzero(as_tuple=True)
            mesa_parameter = self.mesa_module(task_id)
            base_state_diff = self.meta_module(mesa_parameter)
            y_task = self.base_module(x[task_rows], base_state_diff)
            y_parts.append((y_task, task_rows))
        y_parts.sort(key=lambda x: x[1][0])  # Sort by task rows
        y = torch.cat([part[0] for part in y_parts])
        return y

    class MesaModel(torch.nn.Module):
        def __init__(self, mtms):
            super(MtMs.MesaModel, self).__init__()
            self.base_module = mtms.base_module
            self.meta_module = mtms.meta_module
            for param in self.meta_module.parameters():
                param.requires_grad = False
            self.mesa_parameter = torch.nn.Parameter(torch.zeros(mtms.mesa_module.mesa_parameter_size))

        def forward(self, x):
            mesa_parameter = self.mesa_parameter
            base_state_diff = self.meta_module(mesa_parameter)
            y = self.base_module(x, base_state_diff)
            return y
