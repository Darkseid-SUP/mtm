import torch

class MesaModule(torch.nn.Module):
    def __init__(self, num_tasks, mesa_parameter_size, mesa_parameters_init=None):
        super(MesaModule, self).__init__()

        self.mesa_parameter_size = mesa_parameter_size
        self.num_tasks = num_tasks

        if mesa_parameters_init is None:
            INIT_RANGE = 0
            mesa_parameters_init = torch.rand((self.mesa_parameter_size, self.num_tasks)) * 2 * INIT_RANGE - INIT_RANGE
        else:
            assert mesa_parameters_init.shape == (self.mesa_parameter_size, self.num_tasks), "invalid supplied mesa_parameters_init"

        self.mesa_parameters = torch.nn.Parameter(mesa_parameters_init)

    def forward(self, task_id):
        mesa_parameter = self.mesa_parameters[:, task_id]
        return mesa_parameter