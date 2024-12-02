import torch

from contrastive.args import Arguments

class Gate(torch.nn.Module):
    def __init__(self, size):
        super(Gate, self).__init__()
        self.gate = torch.nn.Parameter(torch.ones(size))

    def forward(self, x):
        neuron_to_enable = torch.sigmoid(self.gate)
        return x * neuron_to_enable


class ContrastiveMLP(torch.nn.Module):

    def __init__(self, args: Arguments):

        super(ContrastiveMLP, self).__init__()
        sizes = args.contrastive_mlp_sizes
        if args.input_tokens == "start_end_pair":
            sizes[0] = sizes[0] * 2

        gate = Gate(sizes[0])

        if args.activation == "silu":
            activation = torch.nn.SiLU()
        elif args.activation == "leaky_relu":
            activation = torch.nn.LeakyReLU()
        else:
            activation = torch.nn.ReLU()

        if args.noise == "dropout":
            noise = torch.nn.Dropout(args.dropout)
        else:
            noise = torch.nn.Identity()

        if args.is_hidden_layer:
            assert len(sizes) == 3, "Hidden layer requires 3 sizes"
            input_size, hidden_size, output_size = sizes
            self.net = torch.nn.Sequential(
                gate,
                torch.nn.Linear(input_size, hidden_size),
                activation,
                noise,
                torch.nn.Linear(hidden_size, output_size)
            )

        else:
            assert len(sizes) == 2, "Output layer requires 2 sizes"
            input_size, output_size = sizes
            self.net = torch.nn.Sequential(
                gate,
                activation,
                noise,
                torch.nn.Linear(input_size, output_size)
            )

    def forward(self, x):
        return self.net(x)

class Detector(torch.nn.Module):

    def __init__(self):
        super(Detector, self).__init__()
        self.threshold = torch.nn.Linear(1, 2, bias=True)

    def forward(self, x1, x2):
        cosine = torch.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        return self.threshold(cosine)
