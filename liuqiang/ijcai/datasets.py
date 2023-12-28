import numpy as np
import mindspore as ms
from mindspore_gl.graph.ops import BatchHomoGraph, PadArray2d, PadHomoGraph, PadMode, PadDirection
from mindspore_gl import BatchedGraphField
from mindspore_gl.dataloader import Dataset

class MultiHomoGraphDataset(Dataset):
    """MultiHomoGraph Dataset"""
    def __init__(self, dataset, batch_size, length, mode=PadMode.AUTO, node_size=400, edge_size=2000):
        self._dataset = dataset
        self._batch_size = batch_size
        self._length = length
        self.batch_fn = BatchHomoGraph()
        self.node_size = node_size
  
        if mode == PadMode.CONST:
            self.node_feat_pad_op = PadArray2d(dtype=np.float32, mode=PadMode.CONST, direction=PadDirection.COL,
                                               size=(node_size, dataset.node_feat_size), fill_value=0)
            
            self.graph_pad_op = PadHomoGraph(n_edge=edge_size, n_node=node_size, mode=PadMode.CONST)
        else:
            self.node_feat_pad_op = PadArray2d(dtype=np.float32, mode=PadMode.AUTO, direction=PadDirection.COL,
                                               fill_value=0)

            self.graph_pad_op = PadHomoGraph(mode=PadMode.AUTO)

        # For Padding
        self.train_mask = np.array([True] * (self._batch_size + 1))
        self.train_mask[-1] = False

    def __getitem__(self, batch_graph_idx):
        graph_list = []
        feature_list = []
        for idx in range(batch_graph_idx.shape[0]):
            graph_list.append(self._dataset[batch_graph_idx[idx]])
            feature_list.append(self._dataset.graph_node_feat(batch_graph_idx[idx]))

        # Batch Graph
        batch_graph = self.batch_fn(graph_list)

        # Pad Graph
        batch_graph = self.graph_pad_op(batch_graph)

        # Batch Node Feat
        batched_node_feat = np.concatenate(feature_list)

        # Pad NodeFeat
        batched_node_feat = self.node_feat_pad_op(batched_node_feat)
        batched_label = self._dataset.graph_label[batch_graph_idx]

        # Pad Label
        batched_label = np.append(batched_label, batched_label[-1] * 0)


        # Trigger Node_Map_Idx/Edge_Map_Idx Computation, Because It Is Lazily Computed
        _ = batch_graph.batch_meta.node_map_idx
        _ = batch_graph.batch_meta.edge_map_idx
        
        node_loop = np.arange(0, self.node_size, dtype=np.int32)
        np_graph_mask = [1] * (self._batch_size + 1)
        np_graph_mask[-1] = 0
        constant_graph_mask = ms.Tensor(np_graph_mask, dtype=ms.int32)
        batchedgraphfiled = self.get_batched_graph_field(batch_graph, node_loop, constant_graph_mask)
        row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask = batchedgraphfiled.get_batched_graph()
        batched_label = batched_label.astype('int32')
        batched_node_feat = batched_node_feat.astype('float32')
        return row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask, batched_label,\
               batched_node_feat

    def get_batched_graph_field(self, batch_graph, node_loop, constant_graph_mask):
        return BatchedGraphField(
            ms.Tensor.from_numpy(np.concatenate((batch_graph.adj_coo[0], node_loop))),
            ms.Tensor.from_numpy(np.concatenate((batch_graph.adj_coo[1], node_loop))),
            ms.Tensor(batch_graph.node_count, ms.int32),
            ms.Tensor(batch_graph.edge_count + batch_graph.node_count, ms.int32),
            ms.Tensor.from_numpy(batch_graph.batch_meta.node_map_idx),
            ms.Tensor.from_numpy(
                np.concatenate((batch_graph.batch_meta.edge_map_idx, batch_graph.batch_meta.node_map_idx))),
            constant_graph_mask
        )

    def __len__(self):
        return self._length