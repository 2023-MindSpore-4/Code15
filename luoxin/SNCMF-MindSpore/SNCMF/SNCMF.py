import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score, v_measure_score
from mindspore import Tensor, nn, Parameter, ops, context
from sklearn.preprocessing import normalize
from mindspore.ops import operations as P
import mindspore


context.set_context(device_target="CPU")

def replace_nan_with_zero(input_data):
    # matrix = mindspore.Tensor(np.array(input_data, dtype=np.float32))
    matrix = mindspore.Tensor(np.nan_to_num(input_data.asnumpy(), nan=0))
    return matrix

# NMF模型
class NMF(nn.Cell):
    # 模型初始化
    def __init__(self, num_nodes, n_components, alpha, lamda):
        super(NMF, self).__init__()
        self.alpha = alpha    # 不需要用parameter, parameter为需更新的模型参数
        self.lamda = lamda
        self.n_components = n_components
        self.num_nodes = num_nodes
        np.random.seed(22)  # 22 / 12
        self.U = Tensor((np.abs(np.random.normal(scale=1. / n_components, size=(num_nodes, n_components)))), mindspore.float32)
        self.X = Tensor((np.abs(np.random.normal(scale=1. / n_components, size=(num_nodes, n_components)))), mindspore.float32)
        self.Y = Tensor((np.abs(np.random.normal(scale=1. / n_components, size=(num_nodes, n_components)))), mindspore.float32)
        self.transpose = ops.Transpose()
        self.perm = (1, 0)
        self.norm = P.L2Normalize(axis=0)

    # 模型更新
    def construct(self, adjacency_matrix, D):
        # 将原始的numpy数组转换为MindSpore的Tensor
        AY = adjacency_matrix @ self.Y
        numerator = AY + self.alpha * self.U
        denominator = self.X @ self.transpose(self.Y, self.perm) @ self.Y + self.alpha * self.X
        self.X *= replace_nan_with_zero(numerator / denominator)
        # self.X *= numerator / denominator

        ATX = self.transpose(adjacency_matrix, self.perm) @ self.X
        numerator = ATX + self.alpha * self.U
        denominator = self.Y @ self.transpose(self.X, self.perm) @ self.X + self.alpha * self.Y
        self.Y *= replace_nan_with_zero(numerator / denominator)
        # self.Y *= numerator / denominator

        numerator = self.alpha * (self.X + self.Y) + self.lamda * adjacency_matrix @ self.U
        denominator = 2 * self.alpha * self.U + self.lamda * D @ self.U
        self.U *= replace_nan_with_zero(numerator / denominator)  # 更新U
        # self.U *= numerator / denominator  # 更新U

        return self.U, self.X, self.Y

# 读取数据集
file_path = './processed_dataset.txt'
ground_truth_file = './ground_truth.txt'
data = np.loadtxt(file_path, dtype=int)

# 创建邻接矩阵
num_nodes = np.max(data)
adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
for edge in data:
    adjacency_matrix[edge[0] - 1, edge[1] - 1] = 1

# 创建图正则矩阵
D = np.diag(np.sum(adjacency_matrix, axis=1))

# 模型超参数设置
n_components = 13
lamda = 2e-8  # 图正则的权重参数
max_iterations = 500  # 迭代次数
tolerance = 1e-4  # 收敛容忍度
alpha = 1
pre_rmse = float('inf')
count = 0

model = NMF(num_nodes, n_components, alpha, lamda)

# Converting numpy arrays to MindSpore Tensors
adjacency_matrix = Tensor(adjacency_matrix, dtype=mindspore.float32)
D = Tensor(D, dtype=mindspore.float32)

# 模型训练
for iteration in range(max_iterations):
    U, X, Y = model(adjacency_matrix, D)
    predicted_adjacency = np.dot(X.asnumpy(), Y.asnumpy().T)
    cur_rmse = np.sqrt(np.mean((adjacency_matrix.asnumpy() - predicted_adjacency) ** 2))
    print("epoch %d: %.6f" % (iteration, cur_rmse))
    if pre_rmse - cur_rmse <= tolerance:
        count += 1
        if count == 5:
            break
    pre_rmse = cur_rmse

# Converting MindSpore Tensors to numpy arrays
U = U.asnumpy()
X = X.asnumpy()
Y = Y.asnumpy()

# 归一化W和H以便可视化
# X = normalize(X, norm='l2', axis=1)
# Y = normalize(Y, norm='l2', axis=1)
U = normalize(U, norm='l2', axis=1)

# 绘制结果
community_assignments = np.argmax(U, axis=1)
color_map = ['b' if community_assignments[i] == 0 else 'r' for i in range(num_nodes)]

# plt.scatter(X[:, 0], X[:, 1], c=color_map)
# plt.title("Community Detection with NMF using Lee-Seung Rule")
# plt.xlabel("Community 1")
# plt.ylabel("Community 2")
# plt.show()

# 输出社区分类结果
for i in range(n_components):
    nodes_in_community = np.where(community_assignments == i)[0]
    print(f"Community {i + 1}: {nodes_in_community}")

# 读取真实社区分类结果
with open(ground_truth_file, 'r') as file:
    lines = file.readlines()

ground_truth = []
for index, line in enumerate(lines):
    data = list(map(int, line.split()))
    max_value = max(data)
    if max_value >= len(ground_truth):
        ground_truth.extend([0] * (max_value - len(ground_truth)))
    for idx in data:
        ground_truth[idx-1] = index

# 计算指标
nmi = normalized_mutual_info_score(ground_truth, community_assignments)
print(f"NMI: {nmi:.4f}")

