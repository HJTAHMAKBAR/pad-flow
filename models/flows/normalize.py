"""
Code modified from https://github.com/unknownue/puflow
"""

import torch
import torch.nn as nn

from torch import Tensor


class ActNorm(nn.Module):

    def __init__(self, channel: int, dim=1):
        super(ActNorm, self).__init__()

        assert dim in [-1, 1, 2]
        self.dim = 2 if dim == -1 else dim

        # 根据通道所在的维度初始化scale和bias

        if self.dim == 1:
            self.logs = nn.Parameter(torch.zeros((1, channel, 1)))  # log sigma
            self.bias = nn.Parameter(torch.zeros((1, channel, 1)))
            self.Ndim = 2
        if self.dim == 2:
            self.logs = nn.Parameter(torch.zeros((1, 1, channel)))
            self.bias = nn.Parameter(torch.zeros((1, 1, channel)))
            self.Ndim = 1

        self.eps = 1e-6
        self.is_inited = False

    def forward(self, x: Tensor, _: Tensor = None):
        if not self.is_inited:
            self.__initialize(x)

        z = (x + self.bias) * torch.exp(self.logs)
        # z = x * torch.exp(self.logs) + self.bias      # pu-flow，感觉有问题，改动
        # z = (x - self.bias) * torch.exp(-self.logs)
        logdet = torch.sum(self.logs) * x.shape[self.Ndim]  # 对点云中的所有点（例：1024）执行相同操作，Ndim点数相当于图像中的h,w。
        # 这里返回的logdet对数行列式是标量
        # 后面对每个batch再进行操作
        return z, logdet

    def inverse(self, z: Tensor, _: Tensor = None):
        x = z * torch.exp(-self.logs) - self.bias
        # x = (z - self.bias) * torch.exp(-self.logs)   # pu-flow，感觉有问题，改动
        # x = z * torch.exp(self.logs) + self.bias
        logdet = -torch.sum(self.logs) * x.shape[self.Ndim]     # 逆向过程不需要计算似然，按下不表
        return x, logdet

    def __initialize(self, x: Tensor):
        # data dependent initialization
        # 初始化的目的是为了使得进入到后面网络的输入在单位通道上具有零均值和单位方差
        # 只在第一步进行初始化，根据第一个batch的数据对scale和bias进行初始化，是数据相关的。后面变成可训练的参数。
        with torch.no_grad():
            dims = [0, 1, 2]
            dims.remove(self.dim)

            # 在通道上进行归一化，所以要保留通道维度，对剩余其他维度计算均值和方差。
            # 这里计算的是第一个minibatch在通道维度上的均值和方差
            bias = -torch.mean(x.detach(), dim=dims, keepdim=True)
            logs = -torch.log(torch.std(x.detach(), dim=dims, keepdim=True) + self.eps)     # 标准差std保证是非负
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.is_inited = True


if __name__ == '__main__':
    actnorm = ActNorm(3, dim=2)
    x = torch.randn(2, 10, 3)   # (B, N, C)
    z, logdet = actnorm.forward(x)
    print(z.shape, logdet)
    x, logdet = actnorm.inverse(z)
    print(x.shape, logdet)
