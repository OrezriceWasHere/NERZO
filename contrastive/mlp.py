import torch

from contrastive.args import Arguments


class ContrastiveMLP(torch.nn.Module):

    def __init__(self, args: Arguments):

        super(ContrastiveMLP, self).__init__()

        input_size = self.__calc_input_size(args)
        gate = Gate(input_size)
        activation = self.__build_activation(args)
        noise = self.__build_noise(args)
        middle_layer = self.__build_middle_layer(input_size, args)
        output_layer = self.__build_output_layer(input_size, args)

        self.net = torch.nn.Sequential(
            gate,
            middle_layer,
            activation,
            noise,
            output_layer
        )

        print("created mlp model: ", self.net)

    def forward(self, x):
        return self.net(x)

    def __calc_input_size(self, args: Arguments):
        if args.input_tokens == "start_end_pair":
            return args.input_layer * 2
        return args.input_layer

    def __build_activation(self, args: Arguments):
        if args.activation == "silu":
            return torch.nn.SiLU()
        elif args.activation == "leaky_relu":
            return torch.nn.LeakyReLU()
        return torch.nn.ReLU()

    def __build_noise(self, args: Arguments):
        if args.noise == "dropout":
            return torch.nn.Dropout(args.dropout)
        return torch.nn.Identity()

    def __build_middle_layer(self, input_layer:int, args: Arguments):
        if args.is_hidden_layer:
            return torch.nn.Linear(input_layer, args.hidden_layer)
        return torch.nn.Identity()

    def __build_output_layer(self, input_layer:int,  args: Arguments):
        if args.is_hidden_layer:
            return torch.nn.Linear(args.hidden_layer, args.output_layer)
        return torch.nn.Linear(input_layer, args.output_layer)


class Gate(torch.nn.Module):
    def __init__(self, size):
        super(Gate, self).__init__()
        self.gate = torch.nn.Parameter(torch.ones(size))

    def forward(self, x):
        neuron_to_enable = torch.sigmoid(self.gate)
        return x * neuron_to_enable


class Detector(torch.nn.Module):

    def __init__(self):
        super(Detector, self).__init__()
        self.threshold = torch.nn.Linear(1, 2, bias=True)

    def forward(self, x1, x2):
        cosine = torch.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        return self.threshold(cosine)


