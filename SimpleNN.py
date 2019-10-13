import torch.nn as nn


class SimpleNN(nn.Sequential):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.nn1 = nn.Linear(input_size, hidden_sizes[0])
        nn.ReLU(),

    nn.Linear(*hidden_sizes),
    nn.ReLU(),
    nn.Linear(hidden_sizes[1], output_size),
    nn.LogSoftmax(dim=1)
