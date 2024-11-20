import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, loss_fn="triplet_loss", **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.kwargs = kwargs

        if loss_fn == "triplet_loss":
            self.loss = triplet_loss

        elif loss_fn == "dpr_loss":
            self.loss = dpr_loss


        else:
            raise ValueError(f"Unknown loss function {loss_fn}")

    def forward(self, *args):
        return self.loss(*args, **self.kwargs)


__triplet_loss_criterion = lambda x, y: 1.0 - F.cosine_similarity(x, y, dim=-1)

def triplet_loss(query, positive_key, negative_keys, margin=0.5):
    return F.triplet_margin_with_distance_loss(query, positive_key, negative_keys,
                                               distance_function=__triplet_loss_criterion,
                                               margin=margin)

__dpr_loss_criteria = lambda x, y: F.cosine_similarity(x, y, dim=-1)

def dpr_loss(query, positive_key, negative_keys, **kwargs):
    sum_pos_examples = __dpr_loss_criteria(query, positive_key).exp().sum()
    sum_neg_examples = __dpr_loss_criteria(query, negative_keys).exp().sum()
    return -torch.log(sum_pos_examples / (sum_pos_examples + sum_neg_examples))