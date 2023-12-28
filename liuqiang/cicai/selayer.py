import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore_gl import BatchedGraph
from mindspore_gl.nn import GNNCell

class SELayer(GNNCell):
    def __init__(self, channel, reduction=2):
        super().__init__()

        self.fc = nn.SequentialCell(
            nn.Dense(channel, channel // reduction, has_bias=False),
            nn.ReLU(),
            nn.Dense(channel // reduction, channel, has_bias=False),
            nn.Sigmoid()
        )

    
    def construct(self, x, bg: BatchedGraph):
        x_avg = bg.avg_nodes(x)
        
        feat = self.fc(x_avg)
        b_feat = bg.broadcast_nodes(feat)
        y = ops.mul(x ,b_feat)
        return y
    

class Att_Node(GNNCell):
    def __init__(self, maxnodes=396, reduction=3, batch_size=16):
        super().__init__()
        self.fc1 = nn.SequentialCell(
            nn.Dense(maxnodes, maxnodes // reduction, has_bias=False),
            nn.ReLU(),
            nn.Dense(maxnodes // reduction, maxnodes, has_bias=False),
            nn.Sigmoid()
        )
        self.maxnodes = maxnodes
        self.batch_size = batch_size + 1

    
    def construct(self, x, bg: BatchedGraph):
        batch_num_objs = ops.squeeze(bg.num_of_nodes()) # [3,2,4]

        batch_x = ops.zeros([self.batch_size, self.maxnodes, ops.shape(x)[-1]])
        base_index = 0

        for i in range(self.batch_size-1):
            batch_x[i,:batch_num_objs[i],:] = x[base_index:base_index+batch_num_objs[i],:]
            base_index += batch_num_objs[i]
        

        y = batch_x.mean(axis=-1).unsqueeze(dim=-1)
        y = ops.transpose(y, (0,2,1))
        y = self.fc1(y)


        mask_node = ops.zeros([self.batch_size, 1, self.maxnodes], dtype = ms.bool_)
        for i in range(self.batch_size - 1):
            mask_node[i,0,:batch_num_objs[i]] = 1

        y.masked_fill(mask_node==0, ms.Tensor(-1e9))
        y = ops.softmax(y)

        y = ops.transpose(y, (0,2,1)) * batch_x

        feat_y = []
        for i in range(self.batch_size):
            feat_y.extend(y[i, :batch_num_objs[i]].unsqueeze(dim=0))
        feat_y = ops.cat(feat_y)

        return feat_y