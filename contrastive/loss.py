from random import randint

import torch
import torch.nn.functional as F
from info_nce import InfoNCE


class ContrastiveLoss(torch.nn.Module):


	def __init__(self, loss_fn="triplet_loss", **kwargs):
		super(ContrastiveLoss, self).__init__()
		self.kwargs = kwargs

		if loss_fn == "triplet_loss":
			self.loss = triplet_loss

		elif loss_fn == "dpr_loss":
			# self.loss = dpr_loss
			self.loss = dpr_loss

		elif loss_fn == "contrastive_loss":
			self.loss = contrastive_loss

		elif loss_fn == "triplet_contrastive_loss":
			self.loss = triplet_contrastive_loss



		else:
			raise ValueError(f"Unknown loss function {loss_fn}")

	def forward(self, *args, **kwargs):
		kwargs = {**self.kwargs, **kwargs}
		return self.loss(*args, **kwargs)


def hardest_negative_triplet(query, positive_key, negative_keys, margin):
	arg_hardest_negative = torch.argmax(__triplet_loss_criterion(query, negative_keys))
	hardest_negative = negative_keys[arg_hardest_negative.item()]
	possible_positive = positive_key[randint(0, len(positive_key) - 1)]
	return F.triplet_margin_with_distance_loss(
		query.squeeze(0), possible_positive, hardest_negative,
		distance_function=__triplet_loss_criterion,
		margin=margin
		)


__triplet_loss_criterion = lambda x, y: 1.0 - F.cosine_similarity(x, y, dim=-1)


def triplet_loss(query, positive_key, negative_keys, margin=0.5, epoch=0):
	triplet_loss = F.triplet_margin_with_distance_loss(
		query, positive_key, negative_keys,
		distance_function=__triplet_loss_criterion,
		margin=margin
		)
	hardest_triplet_loss = hardest_negative_triplet(query, positive_key, negative_keys, margin)
	return (max(epoch, 100) /100) * triplet_loss + ((100 - min(epoch, 100)) /100) * hardest_triplet_loss




__dpr_loss_criteria = InfoNCE()


def dpr_loss(queries, positive_keys, negative_keys, temperature=0.07, **kwargs):
	"""
	Computes contrastive loss with multiple positives per query.

	Parameters:
	- queries: Tensor of shape (B, D), batch of query embeddings
	- positives: Tensor of shape (B, P, D), multiple positive embeddings per query
	- negatives: Tensor of shape (B, N, D), negative embeddings per query
	- temperature: Scaling factor for contrastive loss

	Returns:
	- Contrastive loss value (scalar)
	"""

	# Compute similarity scores
	# Compute similarity for all positives (batch-wise)
	sim_pos = F.cosine_similarity(queries, positive_keys)  # (B, P)

	# Compute similarity for all negatives
	sim_neg = F.cosine_similarity(queries, negative_keys)  # (B, N)

	# Apply temperature scaling
	sim_pos = sim_pos / temperature
	sim_neg = sim_neg / temperature

	# Compute log-sum-exp denominator (sum over positives and negatives)
	loss = -torch.log(
		torch.sum(torch.exp(sim_pos)) /
		(torch.sum(torch.exp(sim_pos)) + torch.sum(torch.exp(sim_neg)))
	)

	return loss.mean()


__contrastive_loss_criteria = lambda x, y: F.cosine_similarity(x, y, dim=-1)


def contrastive_loss_fn(x1, x2, label, margin: float = 1.0):
	"""
	Computes Contrastive Loss
	"""

	dist = __contrastive_loss_criteria(x1, x2)

	loss = (1 - label) * torch.pow(dist, 2) \
	       + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
	loss = torch.mean(loss)

	return loss


def contrastive_loss(query, positive_key, negative_keys, margin: float = 1.0):
	positive_label = torch.tensor([1] * len(query)).to(query.device)
	negative_label = torch.tensor([0] * len(query)).to(query.device)
	loss_positive = contrastive_loss_fn(query, positive_key, positive_label)
	loss_negative = contrastive_loss_fn(query, negative_keys, negative_label)
	return loss_positive + loss_negative


def triplet_contrastive_loss(query, positive_key, negative_keys, margin=0.5):
	triplet = triplet_loss(query, positive_key, negative_keys, margin)
	contrastive = contrastive_loss(query, positive_key, negative_keys, margin=margin)
	return triplet + contrastive


def center_loss(query, positive_key, negative_keys, **kwargs)-> torch.Tensor:
	positive_contribution = torch.exp(torch.mean(torch.sum((positive_key - query.squeeze(1)) ** 2, dim=1)))
	negative_contribution = torch.exp(-1 * torch.mean(torch.sum((negative_keys - query.squeeze(1)) ** 2, dim=1)))
	loss = 0.5 * (positive_contribution + negative_contribution)
	return loss.mean()
