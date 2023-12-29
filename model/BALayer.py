import networkx as nx
import torch
import torch.nn as nn
import random

class CustomConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CustomConvolutionalLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # 初始化权重和偏置
        self.weight = nn.Parameter(torch.FloatTensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

        m = 2  # BA模型参数，每个新节点要连接的边数
        G = nx.barabasi_albert_graph(out_channels, m)
        connections = [[[0 for _ in range(kernel_size)] for _ in range(kernel_size)] for _ in range(in_channels)]
        # 根据BA模型设置连接
        for j in range(out_channels):
            for i in G.neighbors(j):
                connections[i][j] = [[1 for _ in range(kernel_size)] for _ in range(kernel_size)]

        # 确保每个输入通道至少有一个连接
        for i in range(in_channels):
            if all(all(value == 0 for row in connections[i]) for value in connections[i]):
                # 随机选择一个输出通道进行连接
                j = random.choice(range(out_channels))
                connections[i][j] = [[1 for _ in range(kernel_size)] for _ in range(kernel_size)]

        # 根据BA模型确定的连接调整权重
        self.mask = torch.FloatTensor(connections)
        self.weight.data *= self.mask

    def forward(self, x):
        weight = self.weight * self.mask
        return nn.functional.conv2d(x, weight, bias=self.bias)