import numpy as np
from mindspore import Tensor
from . import pdist, register_batch_miner


@register_batch_miner
class SoftHard:
	def __init__(self, opt):
		self.opt = opt

	def __call__(self, batch, labels, return_distances=False):
		if isinstance(labels, Tensor):
			# labels = labels.detach().numpy()
			labels = labels.asnumpy()
		bs = batch.shape[0]
		# Return distance matrix for all elements in batch (BSxBS)
		# distances = pdist(batch.detach()).detach().cpu().numpy()
		distances = pdist(batch).asnumpy()

		positives, negatives = [], []
		anchors = []
		for i in range(bs):
			l, d = labels[i], distances[i]
			neg = labels != l
			pos = labels == l

			if np.sum(pos) > 1:
				anchors.append(i)
				# 1 for batch elements with label l
				# 0 for current anchor
				pos[i] = False

				# Find negatives that violate triplet constraint in a hard fashion
				neg_mask = np.logical_and(neg, d < d[np.where(pos)[0]].max())
				# Find positives that violate triplet constraint in a hard fashion
				pos_mask = np.logical_and(pos, d > d[np.where(neg)[0]].min())

				if pos_mask.sum() > 0:
					positives.append(np.random.choice(np.where(pos_mask)[0]))
				else:
					positives.append(np.random.choice(np.where(pos)[0]))

				if neg_mask.sum() > 0:
					negatives.append(np.random.choice(np.where(neg_mask)[0]))
				else:
					negatives.append(np.random.choice(np.where(neg)[0]))

		sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]
		if return_distances:
			return sampled_triplets, distances
		else:
			return sampled_triplets




