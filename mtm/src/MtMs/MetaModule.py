import torch

class MetaModule(torch.nn.Module):
    def __init__(self, base_model, mesa_parameter_size=1, meta_bias=True, meta_dropout_p=0, meta_weight_init=None, meta_connections=None):
        super(MetaModule, self).__init__()

        self.mesa_parameter_size = mesa_parameter_size
        self.state_size = base_model.state_size

        if meta_connections is None:
            meta_connections = list(base_model.state_structure.keys())
        else:
            assert all(x in base_model.state_structure for x in meta_connections), "invalid supplied meta_connections"

        connections = {k: torch.tensor(False) for k in base_model.state_structure.keys()}
        for k in meta_connections:
            connections[k] = torch.tensor(True)

        self.connections = base_model.vectorize_state(connections)

        if meta_weight_init is None:
            INIT_RANGE = 1
            meta_weight_init = torch.rand((self.connections.sum(), self.mesa_parameter_size)) * 2 * INIT_RANGE - INIT_RANGE
        else:
            assert meta_weight_init.shape == (self.connections.sum(), self.mesa_parameter_size), "invalid supplied meta_weight_init"

        self.meta_weight = torch.nn.Parameter(meta_weight_init)

        if meta_bias:
            self.meta_bias = torch.nn.Parameter(torch.zeros(self.state_size))
        
        self.dropout = torch.nn.Dropout(p=meta_dropout_p)

    def forward(self, mesa_parameter):
        base_state_diff = torch.zeros(self.state_size)
        base_state_diff[self.connections.bool()] = torch.matmul(mesa_parameter, self.meta_weight)
        if hasattr(self, 'meta_bias'):
            base_state_diff += self.meta_bias
        base_state_diff = self.dropout(base_state_diff)
        return base_state_diff
