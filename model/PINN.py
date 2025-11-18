import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, input_dim,output_dim, hidden_dim=64, num_layers=2):
        super(PINN, self).__init__()

        layers = []

        layers.append(nn.Linear(2, hidden_dim))
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)