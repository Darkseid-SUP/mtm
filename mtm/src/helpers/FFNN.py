import torch.nn as nn

class FFNN(nn.Module):
    def __init__(self, input_size, layer_sizes, layer_transforms, layer_biases=None, layer_dropout_ps=None, skip_layer=False):
        super(FFNN, self).__init__()
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.layer_transforms = layer_transforms
        layer_sizes_combined = [input_size] + layer_sizes

        if layer_biases is None:
            layer_biases = [True] * len(self.layer_sizes)
        else:
            if not len(layer_biases) == len(self.layer_sizes):
                raise ValueError("invalid layer_biases parameter")

        if layer_dropout_ps is None:
            layer_dropout_ps = [0] * len(self.layer_sizes)
        else:
            if not len(layer_dropout_ps) == len(self.layer_sizes):
                raise ValueError("invalid layer_dropout_ps parameter")
        self.layer_dropout_ps = layer_dropout_ps

        for i in range(len(self.layer_sizes)):
            setattr(self, f"layer_{i}", nn.Linear(layer_sizes_combined[i], layer_sizes_combined[i + 1], bias=layer_biases[i]))

        if skip_layer:
            self.skip_layer = nn.Linear(layer_sizes_combined[0], self.layer_sizes[-1], bias=True)

    def forward(self, x, horizon=1):
        y = x
        for h in range(1, max(horizon) + 1):
            x = y[:, :self.input_size]

            if hasattr(self, "skip_layer"):
                xskip = self.skip_layer(x)

            for i in range(len(self.layer_sizes)):
                x = self.layer_transforms[i](getattr(self, f"layer_{i}")(x))