import torch
from torch.nn import functional as F

class SwiGLU(torch.nn.Module):
    "from https://github.com/lucidrains/PaLM-pytorch/blob/main/palm_pytorch/palm_pytorch.py"
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


# class SwiGLU(torch.nn.Module):
#
#     def __init__(self, w1, w2, w3) -> None:
#         super().__init__()
#         self.w1 = w1
#         self.w2 = w2
#         self.w3 = w3
#
#     def forward(self, x):
#         x1 = F.linear(x, self.w1.weight)
#         x2 = F.linear(x, self.w2.weight)
#         hidden = F.silu(x1) * x2
#         return F.linear(hidden, self.w3.weight)

class ContrastiveMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 activation="silu",
                 noise="identity",
                 dropout=0):
        super(ContrastiveMLP, self).__init__()
        self.gate = torch.nn.Parameter(torch.ones(input_size))
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)


        if activation == "silu":
            self.activation = torch.nn.SiLU()
        else:
            self.activation = torch.nn.ReLU()

        if noise == "dropout":
            self.noise = torch.nn.Dropout(dropout)
        else:
            self.noise = torch.nn.Identity()

        self.net = torch.nn.Sequential(
            self.fc1,
            self.activation,
            self.noise,
            self.fc2
        )

    def forward(self, x):
        neuron_to_enable = torch.sigmoid(self.gate)
        x = x * neuron_to_enable
        return self.net(x)

class Detector(torch.nn.Module):

    def __init__(self, input_dim):
        super(Detector, self).__init__()
        self.threshold = torch.nn.Linear(2 * input_dim, 2)

    def forward(self, x1, x2):
        return self.threshold(torch.cat([x1, x2], dim=-1))
