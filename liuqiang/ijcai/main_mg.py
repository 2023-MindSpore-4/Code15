"""train and eval"""
import time
import argparse
import numpy as np
import mindspore as ms
from mindspore.profiler import Profiler
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.context as context
import mindspore.dataset as ds
from mindspore_gl.nn.gnn_cell import GNNCell
from mindspore_gl.dataloader import RandomBatchSampler
from mindspore_gl.dataset import Enzymes
from mindspore_gl import BatchedGraph, BatchedGraphField
from gnn_mode import GNN
from datasets import MultiHomoGraphDataset


class LossNet(GNNCell):
    """ LossNet definition """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = ms.nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')

    def construct(self, node_feat, target, g: BatchedGraph):
        predict = self.net(node_feat, g)
        target = ops.Squeeze()(target)
        loss = self.loss_fn(predict, target)
        loss = ops.ReduceSum()(loss * g.graph_mask)
        return loss

def main(arguments):
    if arguments.fuse and arguments.device == "GPU":
        context.set_context(device_target=arguments.device, save_graphs=True, save_graphs_path="./computational_graph/",
                            mode=context.GRAPH_MODE, enable_graph_kernel=True)
    else:
        context.set_context(device_target=arguments.device, mode=context.GRAPH_MODE)

    if arguments.profile:
        ms_profiler = Profiler(subgraph="ALL", is_detail=True, is_show_op_path=False, output_path="./prof_result")


    hidden_dim = 64
    root = "dataset/ENZYMES"
    dataset = Enzymes(root)

    train_batch_sampler = RandomBatchSampler(dataset.train_graphs, batch_size=arguments.batch_size)
    train_multi_graph_dataset = MultiHomoGraphDataset(dataset, arguments.batch_size, len(list(train_batch_sampler)))
    test_batch_sampler = RandomBatchSampler(dataset.val_graphs, batch_size=arguments.batch_size)
    test_multi_graph_dataset = MultiHomoGraphDataset(dataset, arguments.batch_size, len(list(test_batch_sampler)))

    train_dataloader = ds.GeneratorDataset(train_multi_graph_dataset, ['row', 'col', 'node_count', 'edge_count',
                                                                       'node_map_idx', 'edge_map_idx', 'graph_mask',
                                                                       'batched_label', 'batched_node_feat'],
                                           sampler=train_batch_sampler)

    test_dataloader = ds.GeneratorDataset(test_multi_graph_dataset, ['row', 'col', 'node_count', 'edge_count',
                                                                     'node_map_idx', 'edge_map_idx', 'graph_mask',
                                                                     'batched_label', 'batched_node_feat'],
                                          sampler=test_batch_sampler)
    
    np_graph_mask = [1] * (arguments.batch_size + 1)
    np_graph_mask[-1] = 0

    net = GNN(
            num_layer=arguments.num_layers,
            input_dim=dataset.node_feat_size,
            hidden_dim=hidden_dim,
            residual=arguments.residual,
            drop_ratio=arguments.drop_ratio,
            JK=arguments.JK,
            n_tasks=dataset.label_dim,
            num_step_set2set=arguments.num_step_set2set,
            num_layer_set2set=arguments.num_layer_set2set
    )


    learning_rates = nn.piecewise_constant_lr(
        [50, 100, 150, 200, 250, 300, 350], [0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125, 0.00015625])
    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=learning_rates)
    loss = LossNet(net)
    train_net = nn.TrainOneStepCell(loss, optimizer)

    for epoch in range(arguments.epochs):
        start_time = time.time()
        net.set_train(True)
        train_loss = 0
        total_iter = 0
        while True:
            for data in train_dataloader:
                row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask, label, node_feat =\
                    data
                batch_homo = BatchedGraphField(row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask)
                train_loss += train_net(node_feat, label, *batch_homo.get_batched_graph()) /\
                              arguments.batch_size
                total_iter += 1
                if total_iter == arguments.iters_per_epoch:
                    break
            if total_iter == arguments.iters_per_epoch:
                break
        train_loss /= arguments.iters_per_epoch
        net.set_train(False)
        train_count = 0
        for data in train_dataloader:
            row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask, label, node_feat= data
            batch_homo = BatchedGraphField(row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask)
            output = net(node_feat, *batch_homo.get_batched_graph()).asnumpy()
            predict = np.argmax(output, axis=1)
            train_count += np.sum(np.equal(predict, label) * np_graph_mask)
        train_acc = train_count / len(list(train_batch_sampler)) / arguments.batch_size
        end_time = time.time()

        test_count = 0
        for data in test_dataloader:
            row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask, label, node_feat = data
            batch_homo = BatchedGraphField(row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask)
            output = net(node_feat, *batch_homo.get_batched_graph()).asnumpy()
            predict = np.argmax(output, axis=1)
            test_count += np.sum(np.equal(predict, label) * np_graph_mask)

        test_acc = test_count / len(list(test_batch_sampler)) / arguments.batch_size
        print('Epoch {}, Time {:.3f} s, Train loss {}, Train acc {:.5f}, Test acc {:.3f}'.format(epoch,
                                                                                                 end_time - start_time,
                                                                                                 train_loss, train_acc,
                                                                                                 test_acc))
    print(f"check time per epoch {(time.time() - start_time) / arguments.epochs}")
    if arguments.profile:
        ms_profiler.analyse()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DiffPool')
    parser.add_argument('--data_path', dest='data_path', help='Input Dataset path')
    parser.add_argument("--device", type=str, default="CPU", help="which device to use")
    # parser.add_argument("--device_id", type=int, default=0, help="which device id to use")
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers networks')
    parser.add_argument('--residual', type=bool, default=False, help='residual')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='drop_ratio')
    parser.add_argument('--JK', type=str, default="last", help='Different implementations of Jk-concat')
    parser.add_argument('--num_step_set2set', type=int, default=6, help='num_iters')
    parser.add_argument('--num_layer_set2set', type=int, default=3, help='num_layer')
    
    parser.add_argument('--epochs', type=int, default=50, help='training epoch')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of input data')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations during per each epoch')

    parser.add_argument('--profile', type=bool, default=False, help="feature dimension")
    parser.add_argument('--fuse', type=bool, default=False, help="feature dimension")
    args = parser.parse_args()
    main(args)