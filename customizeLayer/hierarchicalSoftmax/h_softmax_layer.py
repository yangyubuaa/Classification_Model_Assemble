import torch
from torch.nn.functional import sigmoid

class HSoftmaxLayer(torch.nn.Module):
    def __init__(self, configs):
        super(HSoftmaxLayer, self).__init__()
        self.configs = configs

        # 生成树结构的神经网络层
        

    def forward(self, input):
        # 遍历树结构神经网络层，将编码向量进行
        pass