import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np
from mindspore import context
import random

context.set_context(device_target="CPU")

def getPrediction(aid, bid, S, D, a, b):
    p_value = 0.0
    for r in range(1, rank + 1):
        p_value += S[aid][r] * D[bid][r]
    p_value += (a[aid] + b[bid])
    return p_value

class MF(nn.Cell):
    # 模型初始化
    def __init__(self, row, col, lamda, rank, ppso):
        super(MF, self).__init__()
        self.lamda = lamda
        self.rank = rank
        self.etaV = np.zeros(ppso)
        self.etaX = np.random.uniform(mineta, maxeta, ppso)
        self.S = Tensor(np.random.uniform(-0.05, 0.05, (row + 1, rank + 1)), mindspore.float32)
        self.D = Tensor(np.random.uniform(-0.05, 0.05, (col + 1, rank + 1)), mindspore.float32)
        self.a = Tensor(np.random.uniform(-0.05, 0.05, row + 1), mindspore.float32)
        self.b = Tensor(np.random.uniform(-0.05, 0.05, col + 1), mindspore.float32)

    # 模型更新
    def construct(self, IterCount_PSO, trainData, trainCount, validData, validCount, ppso):
        pbest = np.inf * np.ones(ppso, dtype=np.float32)
        pbest_value = np.zeros(ppso, dtype=np.float32)
        gbest = np.inf
        gbest_value = 0.0
        for t in range(1, IterCount_PSO + 1):
            fitness = np.ones(ppso)
            for p in range(ppso):
                for trainTuple in trainData:
                    aid = int(trainTuple[0])
                    bid = int(trainTuple[1])
                    valueHat = getPrediction(aid, bid, self.S, self.D, self.a, self.b)
                    err = float(trainTuple[2]) - valueHat
                    for r in range(1, rank + 1):
                        self.S[aid][r] += self.etaX[p] * (err * self.D[bid][r] - self.lamda * self.S[aid][r])
                        self.D[bid][r] += self.etaX[p] * (err * self.S[aid][r] - self.lamda * self.D[bid][r])
                    self.a[aid] += self.etaX[p] * (err - self.lamda * self.a[aid])
                    self.b[bid] += self.etaX[p] * (err - self.lamda * self.b[bid])
                # 评估适应度函数
                for validTuple in validData:
                    aid = int(validTuple[0])
                    bid = int(validTuple[1])
                    valueHat = getPrediction(aid, bid, self.S, self.D, self.a, self.b)
                    fitness[p] += (float(validTuple[2]) - valueHat) ** 2
                fitness[p] = np.sqrt(fitness[p]/validCount)
            for p in range(ppso):
                if fitness[p] < pbest[p]:
                    pbest[p] = fitness[p]
                    pbest_value[p] = self.etaX[p]
                if fitness[p] < gbest:
                    gbest = fitness[p]
                    gbest_value = self.etaX[p]

            for p in range(ppso):
                r1 = np.random.rand()
                r2 = np.random.rand()
                self.etaV[p] =  w * self.etaV[p]+ c * r1 * (pbest_value[p] - ((1 - rho) * self.etaX[p] + rho * self.etaV[p])) + c * r2 * (gbest_value - ((1 - rho) * self.etaX[p] + rho * self.etaV[p]))
                if self.etaV[p] < minV:
                    self.etaV[p] = minV
                if self.etaV[p] > maxV:
                    self.etaV[p] = maxV
                self.etaX[p] += self.etaV[p]
                if self.etaX[p] < mineta:
                    self.etaX[p] = mineta
                if self.etaX[p] > maxeta:
                    self.etaX[p] = maxeta
        return self.S, self.D, self.a, self.b

def evolution(S, D, a, b, testData, testCount):
    RSME = 0
    MAE = 0
    for testTuple in testData:
        aid = int(testTuple[0])
        bid = int(testTuple[1])
        valueHat = getPrediction(aid, bid, S, D, a, b)
        RSME += (float(testTuple[2]) - valueHat) ** 2
        MAE += abs(float(testTuple[2]) - valueHat)

    return np.sqrt(RSME/testCount), MAE/testCount

def SubgroupUpdate(sub, e, beTa, population, lam, M, mark):
    BEST = [0] * (rank_bias + 1)
    P_rank_bias = [[0] * (rank_bias + 1) for _ in range(population)]

    for r in range(1, rank_bias + 1):
        max_val = (1 + beTa) * sub[r]
        min_val = (1 - beTa) * sub[r]
        if max_val < min_val:
            max_val, min_val = min_val, max_val

        P_rank_bias[0][r] = sub[r]
        for p in range(1, population):
            P_rank_bias[p][r] = min_val + random.random() * (max_val - min_val)

    max_a = (1 + beTa) * sub[rank_bias]
    min_a = (1 - beTa) * sub[rank_bias]
    if max_a < min_a:
        max_a, min_a = min_a, max_a

    for p in range(1, population):
        P_rank_bias[p][rank_bias] = min_a + random.random() * (max_a - min_a)
    P_rank_bias[0][rank_bias] = sub[rank_bias]

    best_value = [0] * (rank_bias + 1)
    for r in range(1, rank_bias + 1):
        best_value[r] = P_rank_bias[0][r]
    best_fitness = np.inf

    if mark:
        pre_fitness = [0] * population
        for p in range(population):
            pre_fitness[p] = Calculate_a(P_rank_bias[p], e, lam)
        for round_de in range(1, IterCount_DE + 1):
            for p in range(population):
                dm = [0] * (rank_bias + 1)
                for r in range(1, rank_bias + 1):
                    dm[r] = best_value[r] + M * (P_rank_bias[random.randint(0, population - 1)][r] - P_rank_bias[random.randint(0, population - 1)][r])
                temp = Calculate_a(dm, e, lam)
                if temp < pre_fitness[p]:
                    for r in range(1, rank_bias + 1):
                        P_rank_bias[p][r] = dm[r]
                    pre_fitness[p] = temp
                if pre_fitness[p] < best_fitness:
                    for r in range(1, rank_bias + 1):
                        best_value[r] = P_rank_bias[p][r]
                    best_fitness = pre_fitness[p]
    else:
        pre_fitness = [0] * population
        for p in range(population):
            pre_fitness[p] = Calculate_b(P_rank_bias[p], e, lam)
        for round_de in range(1, IterCount_DE + 1):
            for p in range(population):
                dm = [0] * (rank_bias + 1)
                for r in range(1, rank_bias + 1):
                    dm[r] = best_value[r] + M * (P_rank_bias[random.randint(0, population - 1)][r] - P_rank_bias[random.randint(0, population - 1)][r])
                temp = Calculate_b(dm, e, lam)
                if temp < pre_fitness[p]:
                    for r in range(1, rank_bias + 1):
                        P_rank_bias[p][r] = dm[r]
                    pre_fitness[p] = temp

                if pre_fitness[p] < best_fitness:
                    for r in range(1, rank_bias + 1):
                        best_value[r] = P_rank_bias[p][r]
                    best_fitness = pre_fitness[p]

    for r in range(1, rank_bias + 1):
        BEST[r] = best_value[r]

    return BEST

def Calculate_a(arr, u, lamb):
    temp = 0
    for r in range(1, rank + 1):
        S[u][r] = arr[r]
    a[u] = arr[rank_bias]

    temp_p = 0
    for r in range(1, rank + 1):
        temp_p += S[u][r] ** 2

    count = 0
    for trainTuple in train_data:
        if int(trainTuple[0]) == u:
            count += 1
            temp += (float(trainTuple[2]) - getPrediction(int(trainTuple[0]), int(trainTuple[1]), S, D, a, b)) ** 2

    temp = 0.5 * temp + lamb * 0.5 * (temp_p + a[u] ** 2)

    if count == 0:
        return float('inf')
    else:
        return temp

def Calculate_b(arr, i, lamb):
    temp = 0
    b[i] = arr[rank_bias]

    for r in range(1, rank + 1):
        D[i][r] = arr[r]

    temp_p = 0
    for r in range(1, rank + 1):
        temp_p += D[i][r] ** 2

    count = 0
    for trainTuple in train_data:
        if int(trainTuple[1]) == i:
            count += 1
            temp += (float(trainTuple[2]) - getPrediction(int(trainTuple[0]), int(trainTuple[1]), S, D, a, b)) ** 2

    temp = 0.5 * temp + lamb * 0.5 * (temp_p + b[i] ** 2)

    if count == 0:
        return float('inf')
    else:
        return temp


if __name__ == '__main__':

    DatasetName = "MovieLens-1k"

    # 加载训练、验证和测试数据
    train_data = Tensor(
        np.loadtxt("D:\\PythonWorkplace\\Matrix\\1K\\0.7-0.1-0.2\\ML_train_0.7.txt", delimiter="::", dtype=float),
        dtype=mindspore.float32)
    valid_data = Tensor(
        np.loadtxt("D:\\PythonWorkplace\\Matrix\\1K\\0.7-0.1-0.2\\ML_val_0.1.txt", delimiter="::", dtype=float),
        dtype=mindspore.float32)
    test_data = Tensor(
        np.loadtxt("D:\\PythonWorkplace\\Matrix\\1K\\0.7-0.1-0.2\\ML_test_0.2.txt", delimiter="::", dtype=float),
        dtype=mindspore.float32)

    trainCount = len(train_data)
    validCount = len(valid_data)
    testCount = len(test_data)

    # 数据集不同 参数不同
    maxA = 60  # row
    maxB = 60  # col

    rank = 10
    rank_bias = rank + 1
    lamda = 0.03

    IterCount_PSO = 10
    IterCount_SG = 100
    IterCount_DE = 10

    # 定义PSO超参数
    ppso = 4
    c = 2
    rho = 1
    w = 0.1

    maxeta = 2 ** -8
    mineta = 2 ** -12
    maxV = 1.0
    minV = -1.0

    # 定义de参数
    PDE = 4
    m = 0.1
    beta = 0.1

    fit_RMSE = []
    fit_MAE = []

    fit_RMSE.append(100.0)
    fit_MAE.append(100.0)

    flag_RMSE = True
    flag_MAE = True
    tr = 0
    model = MF(maxA, maxB, lamda, rank, ppso)
    S,D,a,b = model.construct(IterCount_PSO, train_data, trainCount, valid_data, validCount, ppso)
    for t in range(1, IterCount_SG + 1):
        for aid in range(1, maxA + 1):
            mark = True
            subgroup = [0.0] * (rank_bias + 1)
            for r in range(1, rank + 1):
                subgroup[r] = S[aid][r]
            subgroup[rank_bias] = a[aid]
            BEST = SubgroupUpdate(subgroup, aid, beta, PDE, lamda, m, mark)
            for r in range(1, rank + 1):
                S[aid][r] = BEST[r]
            a[aid] = BEST[rank_bias]

        for bid in range(1, maxB + 1):
            mark = False
            subgroup = [0.0] * (rank_bias + 1)
            for r in range(1, rank + 1):
                subgroup[r] = D[bid][r]
            subgroup[rank_bias] = b[bid]
            BEST = SubgroupUpdate(subgroup, bid, beta, PDE, lamda, m, mark)
            for r in range(1, rank + 1):
                D[bid][r] = BEST[r]
            b[bid] = BEST[rank_bias]
        RMSE, MAE = evolution(S, D, a, b, test_data, testCount)
        fit_RMSE.append(RMSE)
        fit_MAE.append(MAE)

        if fit_RMSE[t-1] - fit_RMSE[t] > 0.0001:
            flag_RMSE = False
            tr = 0
        if fit_MAE[t-1] - fit_MAE[t] > 0.0001:
            flag_MAE = False
            tr = 0
        if flag_RMSE and fit_MAE:
            tr += 1
            if tr == 2:
                break

        flag_RMSE = True
        flag_MAE = True

        print("Iteration {}:: test RMSE:: {}:: test MAE:: {}".format(t, RMSE, MAE))
    print("test minRMSE: {} minRMSERound:{}".format(min(fit_RMSE), fit_RMSE.index(min(fit_RMSE))))
    print("test minMAE: {} minMAERound:{}".format(min(fit_MAE), fit_MAE.index(min(fit_MAE))))
    print("所使用的数据集：{}".format(DatasetName))