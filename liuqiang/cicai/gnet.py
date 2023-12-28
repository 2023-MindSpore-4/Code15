import mindspore as ms
import mindspore.nn as nn
from mindspore_gl.nn.gnn_cell import GNNCell
from mindspore_gl.nn.conv import GINConv
from mindspore_gl import BatchedGraph
from selayer import SELayer, Att_Node

class ApplyNodeFunc(nn.Cell):
    """
    Update the node feature hv with MLP.
    """

    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def construct(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = nn.ReLU()(h)
        return h

class MLP(nn.Cell):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
            num_layers: number of layers. If num_layers=1, this reduces to linear model.
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


class SGANNet(GNNCell):
    "GINconv Net"
    def __init__(self, 
                 num_layers,
                 num_mlp_layers,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 final_dropout=0.9,
                 learn_eps=False,
                 graph_pooling_type='sum',
                 neighbor_pooling_type='sum'):
        super().__init__()
        self.final_dropout = final_dropout
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps

        self.mlps = nn.CellList()
        self.convs = nn.CellList()
        self.se = nn.CellList()
        self.att = nn.CellList()
        self.batch_norms = nn.CellList()

        if self.graph_pooling_type not in ('sum', 'avg', 'max'):
            raise SyntaxError("graph pooling type not supported.")
        for layer in range(num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, input_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, input_dim))
            self.convs.append(GINConv(ApplyNodeFunc(self.mlps[layer]), learn_eps=self.learn_eps,
                                      aggregation_type=self.neighbor_pooling_type))
            self.se.append(SELayer(input_dim, 2))
            self.att.append(Att_Node(396, 3))
            

            self.batch_norms.append(nn.BatchNorm1d(input_dim))

        
        self.linears_prediction = nn.CellList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Dense(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Dense(input_dim, output_dim))
        
    def construct(self, x, edge_weight, g: BatchedGraph):
            """construct function"""
            hidden_rep = [x]
            h = x
            for layer in range(self.num_layers - 1):

                h = self.convs[layer](h, edge_weight, g)

                h = self.se[layer](h, g)

                h = self.att[layer](h, g)

                h = self.batch_norms[layer](h)
                h = nn.ReLU()(h)

                hidden_rep.append(h)

            score_over_layer = 0
            for layer, h in enumerate(hidden_rep):
                if self.graph_pooling_type == 'sum':
                    pooled_h = g.sum_nodes(h)
                else:
                    pooled_h = g.avg_nodes(h)

                score_over_layer = score_over_layer + nn.Dropout(p=self.final_dropout)(
                    self.linears_prediction[layer](pooled_h))

            return score_over_layer
    

if __name__ == '__main__':
    pass