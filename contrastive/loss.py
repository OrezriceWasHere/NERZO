import torch
import torch.nn.functional as F
from torch import nn

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, loss_fn="triplet_loss", **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.kwargs = kwargs

        if loss_fn == "triplet_loss":
            self.loss = triplet_loss

        else:
            raise ValueError(f"Unknown loss function {loss_fn}")

    def forward(self, *args):
        return self.loss(*args, **self.kwargs)

def triplet_loss(query, positive_key, negative_keys=None):
    negative_examples = negative_keys[:len(positive_key)]
    return F.triplet_margin_with_distance_loss(query, positive_key, negative_examples, distance_function=nn.PairwiseDistance())
