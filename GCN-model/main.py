import torch
import numpy as np
import pandas as pd
import random
import math
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.utils import remove_self_loops, add_self_loops
from GraphConv import *


def dataset(loc_num, data_onehot, data_density, data_edge):
    """
    定义图结构
    :param data_onehot: 对每个地区进行one hot编码，尺寸是 [城市的区域数*903]
    :param data_density: 将density数据作为模型的标签，尺寸是 [城市的区域数*1]
    :param data_edge: 定义图结构的边信息
    :return: 整个网络的输入图结构信息
    """

    x = torch.from_numpy(data_onehot)
    data_density = data_density.astype(np.float32)
    y = torch.from_numpy(data_density)
    y = y.view(loc_num, 1)
    data_edge = data_edge.astype(np.long)
    edge_index = torch.from_numpy(data_edge)
    data = Data(x=x, y=y, edge_index=edge_index)

    return data


def norm(x):
    """
    对部分特征进行log归一化处理
    :param x: 需要归一化的输入特征
    :return: 归一化后的数据
    """
    return (math.log(x+1) + 2) / 10


def denorm(x):
    """
    对输出的日感染病人数进行恢复
    :param x: 需要去归一化的输出特征
    :return: 日感染病人数(整数)
    """
    return round(math.exp(x * 10 - 2))


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = GraphConv(embed_dim, 128)
        self.lin1 = torch.nn.Linear(128, 32)
        self.lin4 = torch.nn.Linear(32, 8)
        self.lin2 = torch.nn.Linear(9, 4)
        self.lin3 = torch.nn.Linear(4, 1)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.4)

    def forward(self, data, x1, edge_weight, loc_num):
        x, edge_index = data.x, data.edge_index
        x = self.act1(self.conv1(x, edge_index, edge_weight))

        x = self.act1(self.lin1(x))
        x = self.act2(self.lin4(x))
        embedding = x

        x1 = x1.view(loc_num, 1)
        x = torch.cat((x, x1), 1)

        x = self.dropout(self.act2(self.lin2(x)))
        x = self.act1(self.lin3(x))

        return x, embedding


def data_process(arg, city):
    # 这里需要设置训练集和测试集
    # 训练集是用density有值的那部分
    # 测试集用未知density的部分做
    # 目的是获得未知的density，同时会提取网络的中间层embedding做特征

    density = pd.read_csv(arg[0], header=None)
    density = np.array(density.values.tolist())
    density = np.delete(density, 0, axis=1)
    for i in range(np.shape(density)[0] - 1):
        for j in range(np.shape(density)[1]):
            density[i + 1, j] = norm(float(density[i + 1, j]))
    density = density.astype(np.float32)

    infection = pd.read_csv(arg[1], header=None)
    infection = np.array(infection.values.tolist())
    infection = np.delete(infection, 0, axis=1)
    for i in range(np.shape(infection)[0] - 1):
        for j in range(np.shape(infection)[1]):
            infection[i + 1, j] = norm(float(infection[i + 1, j]))
    infection = infection.astype(np.float32)

    edge = pd.read_csv(arg[2], header=None)
    edge = np.array(edge.values.tolist())
    edge = np.delete(edge, 0, axis=0)

    one_hot = pd.read_csv(arg[3], header=None)
    one_hot = np.array(one_hot.values.tolist())
    one_hot = np.delete(one_hot, 0, axis=0)
    one_hot = np.delete(one_hot, 0, axis=1)
    one_hot = one_hot.astype(np.float32)

    data_edge = np.transpose(edge[:, (1, 2)])  # 每天的数据是一致的
    for i in range(np.shape(data_edge)[1]):
        for j in range(2):
            data_edge[j, i] = data_edge[j, i].replace(city, '')
            data_edge[j, i] = data_edge[j, i].replace('_', '')
            data_edge[j, i] = int(data_edge[j, i])

    edge_weight = np.transpose(edge[:, 3])
    for i in range(np.shape(edge)[0]):
        edge_weight[i] = norm(float(edge_weight[i]))

    edge_weight = edge_weight.astype(np.float32)
    edge_weight = torch.from_numpy(edge_weight)  # 每天的数据是一致的

    data_onehot = one_hot  # 每一天的数据也是一致的

    return density, infection, data_edge, edge_weight, data_onehot


def train(arg, city, loc_num):
    density, infection, data_edge, edge_weight, data_onehot = data_process(arg, city)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    crit = torch.nn.MSELoss()  # 用交叉熵函数来计算损失
    # 划分训练集
    # 这里需要按照数据集大小设置循环 epoch 和 batch_size
    batch_size = 4
    epoch = 200
    loss_result = np.zeros((epoch, 1))
    for k in range(epoch):
        loss_all = 0
        for i in range(batch_size):
            index = random.randint(0, np.shape(density)[1] - 1)
            data_density = density[1:, index]
            for j in range(np.shape(infection)[1]):
                if density[0, index] == infection[0, j]:
                    data_infection = infection[1:, index]
            data = dataset(loc_num, data_onehot, data_density, data_edge)
            x1 = torch.from_numpy(data_infection)  # 将每个区域的新增感染病人数作为补充的输入特征，尺寸是[城市的区域数*1]

            data = data.to(device)
            x1 = x1.to(device)
            edge_weight = edge_weight.to(device)
            optimizer.zero_grad()
            output, _ = model(data, x1, edge_weight, loc_num)
            label = data.y.to(device)
            loss = crit(output, label)
            loss.backward()
            optimizer.step()
            loss = loss.item()*loc_num

            loss_all += loss

        loss = loss_all / batch_size
        loss_result[k, 0] = loss
        print('Epoch:', k, 'Loss:', loss)

    np.savetxt(arg[4], loss_result, delimiter=",")
    print('城市', city, '模型已经训练完毕')


def output_data(arg, arg2, loc_num, city):
    density, infection, data_edge, edge_weight, data_onehot = data_process(arg, city)
    density_rst = np.zeros((loc_num, np.shape(infection)[1]))
    embedding_rst = np.zeros((loc_num, 8))
    crit = torch.nn.MSELoss()  # 用交叉熵函数来计算损失
    # 提取embedding
    data_infection = infection[1:, 0]
    data_density = density[1:, 0]  # 该定义不存在实际意义
    data = dataset(loc_num, data_onehot, data_density, data_edge)
    x1 = torch.from_numpy(data_infection)
    data = data.to(device)
    x1 = x1.to(device)
    edge_weight = edge_weight.to(device)
    output, embedding = model(data, x1, edge_weight, loc_num)
    for j in range(loc_num):
        for k in range(8):
            embedding_rst[j, k] = embedding[j, k]
    label = data.y.to(device)
    loss = crit(output, label)
    loss = loss.item() * loc_num

    # 提取density
    for i in range(np.shape(infection)[1]):
        data_infection = infection[1:, i]
        data_density = density[1:, 0]  # 该定义不存在实际意义
        data = dataset(loc_num, data_onehot, data_density, data_edge)
        x1 = torch.from_numpy(data_infection)

        data = data.to(device)
        x1 = x1.to(device)
        edge_weight = edge_weight.to(device)

        output, _ = model(data, x1, edge_weight, loc_num)

        for j in range(loc_num):
            density_rst[j, i] = output[j]

    np.savetxt(arg2[0], embedding_rst, delimiter=",")
    np.savetxt(arg2[1], density_rst, delimiter=",")
    print('城市', city, '数据已经处理完毕', 'loss:', loss)


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embed_dim = 903
    model = Net().to(device)
    model.train()
    city = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    loc_num = [118, 30, 135, 75, 34, 331, 38, 53, 33, 8, 48]
    for m in range(len(city)):
        city_n = city[m]
        arg_in = ['./DATA/city_'+city_n+'/den_rst.csv', './DATA/city_'+city_n+'/inf_rst.csv',
                  './DATA/tranfer_a_day_'+city_n+'.csv', './DATA/city_'+city_n+'/one_hot.csv',
                  './result/city_'+city_n+'_loss.csv']
        train(arg_in, city_n, loc_num[m])

    model.eval()
    for m in range(len(city)):
        arg_in = ['./DATA/city_'+city[m]+'/den_rst.csv', './DATA/city_'+city[m]+'/inf_rst.csv',
                  './DATA/tranfer_a_day_'+city[m]+'.csv', './DATA/city_'+city[m]+'/one_hot.csv',
                  './result/city_'+city[m]+'_loss.csv']
        arg_out = ['./result/city_'+city[m]+'_embedding.csv', './result/city_'+city[m]+'_density.csv']
        city_n = city[m]
        output_data(arg_in, arg_out, loc_num[m], city_n)
