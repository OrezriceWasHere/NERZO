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

        input_layer = args.input_layer
        if args.input_tokens == "start_end_pair":
            input_layer = args.input_layer * 2

        gate = Gate(input_layer)

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
            self.net = torch.nn.Sequential(
                gate,
                torch.nn.Linear(args.input_layer, args.hidden_layer),
                activation,
                noise,
                torch.nn.Linear(args.hidden_layer, args.output_layer)
            )

        else:
            self.net = torch.nn.Sequential(
                gate,
                activation,
                noise,
                torch.nn.Linear(args.input_layer, args.output_layer)
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
