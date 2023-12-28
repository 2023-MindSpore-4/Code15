import mindspore as ms
import mindspore.nn as nn
from mindspore_gl.nn import GNNCell
from mindspore_gl.nn import GCNConv2
from mindspore_gl.nn.glob import Set2Set
from mindspore_gl import BatchedGraph
from mindspore import ops

class MLP(nn.Cell):
    """MLP"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1,
              this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
        """
        super().__init__()
        self.num_layers = num_layers
        self.output_dim = output_dim
        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        if num_layers == 1:
            # Linear model
            self.linear = nn.Dense(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linears = nn.CellList()
            self.batch_norms = nn.CellList()

            self.linears.append(nn.Dense(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Dense(hidden_dim, hidden_dim))
            self.linears.append(nn.Dense(hidden_dim, output_dim))

            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def construct(self, x):
        if self.num_layers == 1:
            return self.linear(x)
        # If MLP
        h = x
        for layer in range(self.num_layers - 1):
            h = ms.ops.ReLU()(self.batch_norms[layer](self.linears[layer](h)))
        return self.linears[self.num_layers - 1](h)


class GNN_node(GNNCell):
    """GNN_node"""
    def __init__(self,
                 num_layers,
                 input_dim,
                 hidden_dim,
                 drop_ratio=0.5,
                 JK = "last",
                 residual = False
                 ):
        super().__init__()
        self.drop_ratio = drop_ratio
        self.num_layers = num_layers
        self.JK = JK
        self.residual = residual

        self.mlps = nn.CellList()
        self.convs = nn.CellList()
        self.batch_norms = nn.CellList()

        for layer in range(num_layers):
            if layer == 0:
                self.convs.append(GCNConv2(input_dim, hidden_dim))
            else:
                self.convs.append(GCNConv2(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def construct(self, x, g: BatchedGraph):
        """construct function"""
        h_list = [x]
        for layer in range(self.num_layers):
            h = self.convs[layer](h_list[layer], g)
            h = self.batch_norms[layer](h)

            if layer == self.num_layers - 1:
                h = ops.dropout(h, self.drop_ratio)


            if self.residual:
                h += h_list[layer]

            h_list.append(h)
        
        if self.JK == "last":
            node_representation = h_list[-1]
        else:
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation

class GNN(GNNCell):
    """GNN Predictor"""
    def __init__(self,
                 num_layer,
                 input_dim,
                 hidden_dim,
                 residual = False,
                 drop_ratio = 0.5,
                 JK = 'last',
                 n_tasks=6,
                 num_step_set2set=6,
                 num_layer_set2set=3):
        super().__init__()

        self.gnn = GNN_node(num_layer,
                            input_dim,
                            hidden_dim,
                            drop_ratio,
                            JK,
                            residual)
        
        self.readout = Set2Set(input_size=hidden_dim,
                               num_iters=num_step_set2set,
                               num_layers=num_layer_set2set)
        
        self.predict = nn.SequentialCell(
            nn.Dense(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dense(hidden_dim, n_tasks)
        )

    def construct(self, node_feats, bg: BatchedGraph):
        node_feats = self.gnn(node_feats, bg)
        graph_feats = self.readout(node_feats, bg)
        return self.predict(graph_feats)


    
if __name__ == "__main__":
    pass