import torch
from collections.abc import Iterable

class BaseModule(torch.nn.Module):
    def __init__(self, base_model):
        super(BaseModule, self).__init__()

        self.state_structure = {k: v.shape for k, v in base_model.state_dict().items()}

        def vectorize_state(state):
            return torch.cat([v.view(-1) for v in state.values()])

        self.vectorize_state = vectorize_state

        def unvectorize_state(state):
            counter = 0
            out = {}
            for k, shape in self.state_structure.items():
                size = torch.prod(torch.tensor(shape)).item()
                out[k] = state[counter:counter+size].view(shape)
                counter += size
            return out

        self.unvectorize_state = unvectorize_state

        vectorization_test = all([torch.allclose(v, base_model.state_dict()[k]) for k, v in self.unvectorize_state(self.vectorize_state(base_model.state_dict())).items()])

        assert vectorization_test, "something went wrong with vectorization of state"

        self.base_state_init = self.vectorize_state(base_model.state_dict()).clone().detach()  # ensure no gradients
        assert hasattr(base_model, 'fforward'), "base_model must by supplied with working fforward function"
        
        self.fforward = base_model.fforward
        self.state_size = self.vectorize_state(base_model.state_dict()).size()
        
    def forward(self, x, base_state_diff):
        base_state_init = torch.tensor(self.base_state_init)
        base_state = base_state_init + base_state_diff
        base_state = self.unvectorize_state(base_state)
        y = self.fforward(x, base_state)
        return y
