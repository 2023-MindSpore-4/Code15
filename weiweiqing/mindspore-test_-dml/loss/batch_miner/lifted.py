import numpy as np
from mindspore import Tensor
from . import register_batch_miner


@register_batch_miner
class Lifted:
    def __init__(self, opt):
        self.opt = opt

    def __call__(self, batch, labels):
        if isinstance(labels, Tensor):
            # labels = labels.detach().cpu().numpy()
            labels = labels.asnumpy()

        anchors, positives, negatives = [], [], []
        list(range(len(batch)))

        for i in range(len(batch)):
            anchor = i
            pos = labels == labels[anchor]

            if np.sum(pos) > 1:
                anchors.append(anchor)
                positive_set = np.where(pos)[0]
                positive_set = positive_set[positive_set != anchor]
                positives.append(positive_set)

        negatives = []
        for anchor, positive_set in zip(anchors, positives):
            neg_idxs = [i for i in range(len(batch)) if i not in [anchor] + list(positive_set)]
            negative_set = np.arange(len(batch))[neg_idxs]
            negatives.append(negative_set)

        return anchors, positives, negatives
