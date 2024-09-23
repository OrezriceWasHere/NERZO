import torch

class ContrastiveMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 activation="relu",
                 noise="dropout",
                 dropout=0.1):
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
        self.threshold = torch.nn.Linear(1, 2, bias=True)

    def forward(self, x1, x2):
        cosine = torch.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        return self.threshold(cosine)
