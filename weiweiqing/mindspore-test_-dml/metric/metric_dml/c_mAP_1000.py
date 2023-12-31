import mindspore
import numpy as np
import faiss
from . import register_metric


@register_metric
class C_mAP_1000:
    def __init__(self, **kwargs):
        self.requires = ['features_cosine', 'target_labels']
        self.name     = 'c_mAP_1000'

    def __call__(self, target_labels, features_cosine):
        labels, freqs = np.unique(target_labels, return_counts=True)
        R             = 1000

        faiss_search_index  = faiss.IndexFlatIP(features_cosine.shape[-1])
        if isinstance(features_cosine, mindspore.Tensor):
            features_cosine = features_cosine.asnumpy()
            res = faiss.StandardGpuResources()
            faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
        faiss_search_index.add(features_cosine)
        nearest_neighbours  = faiss_search_index.search(features_cosine, int(R+1))[1][:,1:]

        target_labels = target_labels.reshape(-1)
        nn_labels = target_labels[nearest_neighbours]

        avg_r_precisions = []
        for label, freq in zip(labels, freqs):
            rows_with_label = np.where(target_labels==label)[0]
            for row in rows_with_label:
                n_recalled_samples           = np.arange(1,R+1)
                target_label_occ_in_row      = nn_labels[row,:]==label
                cumsum_target_label_freq_row = np.cumsum(target_label_occ_in_row)
                avg_r_pr_row = np.sum(cumsum_target_label_freq_row*target_label_occ_in_row/n_recalled_samples)/1000  # freq
                avg_r_precisions.append(avg_r_pr_row)

        return np.mean(avg_r_precisions)
