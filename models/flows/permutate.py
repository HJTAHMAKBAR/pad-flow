import torch
import numpy as np

from torch import nn
from torch import Tensor


class Permutation(nn.Module):

    def __init__(self, permutation: str, n_channel: int, dim: int):
        super(Permutation, self).__init__()

        assert permutation in ['reverse', 'random', 'inv1x1']
        assert dim in [-1, 1, 2, 3]
        if permutation == 'inv1x1':
            self.permutater = InvertibleConv1x1_1D(n_channel, dim)
        else:
            if dim == -1: self.permutater = _ShufflePermutationXD(permutation, n_channel)
            if dim == 1: self.permutater = _ShufflePermutation1D(permutation, n_channel)
            if dim == 2: self.permutater = _ShufflePermutation2D(permutation, n_channel)
            if dim == 3: self.permutater = _ShufflePermutation3D(permutation, n_channel)

    def forward(self, x: Tensor, _c: Tensor):
        return self.permutater(x)

    def inverse(self, z: Tensor, _c: Tensor):
        return self.permutater.inverse(z)


class _ShufflePermutation(nn.Module):

    def __init__(self, permutation: str, n_channel: int):
        super(_ShufflePermutation, self).__init__()

        if permutation == 'reverse':
            direct_idx = np.arange(n_channel - 1, -1, -1).astype(np.longlong)
            inverse_idx = _ShufflePermutation.get_reverse(direct_idx, n_channel)
        if permutation == 'random':
            direct_idx = np.arange(n_channel - 1, -1, -1).astype(np.longlong)
            np.random.shuffle(direct_idx)
            inverse_idx = _ShufflePermutation.get_reverse(direct_idx, n_channel)

        # 不需要反向传播
        self.register_buffer('direct_idx', torch.from_numpy(direct_idx))
        self.register_buffer('inverse_idx', torch.from_numpy(inverse_idx))

    @staticmethod
    def get_reverse(idx, n_channel: int):
        indices_inverse = np.zeros((n_channel,), dtype=np.longlong)
        for i in range(n_channel):
            indices_inverse[idx[i]] = i
        return indices_inverse

    def forward(self, _x):
        raise NotImplementedError()

    def inverse(self, _z):
        raise NotImplementedError()


class _ShufflePermutation1D(_ShufflePermutation):

    def forward(self, x: Tensor):
        return x[:, self.direct_idx], None

    def inverse(self, z: Tensor):
        return z[:, self.inverse_idx], None


class _ShufflePermutation2D(_ShufflePermutation):

    # 对第三个维度（通道维度）进行重新排序
    # 按照idx里的顺序返回数据
    def forward(self, x: Tensor):
        return x[:, :, self.direct_idx], None

    def inverse(self, z: Tensor):
        return z[:, :, self.inverse_idx], None


class _ShufflePermutation3D(_ShufflePermutation):

    def forward(self, x: Tensor):
        return x[:, :, :, self.direct_idx], None

    def inverse(self, z: Tensor):
        return z[:, :, :, self.inverse_idx], None


class _ShufflePermutationXD(_ShufflePermutation):

    def forward(self, x: Tensor):
        return x[..., self.direct_idx], None

    def inverse(self, z: Tensor):
        return z[..., self.inverse_idx], None


# 直接求解行列式，没有使用glow里面的LU分解，因为矩阵只有3x3
class InvertibleConv1x1_1D(nn.Module):

    def __init__(self, channel: int, dim: int):
        super(InvertibleConv1x1_1D, self).__init__()

        # 可逆1x1卷积本质上就是MLP
        w_init = np.random.randn(channel, channel)
        # QR分解可以对任意矩阵使用，得到的Q是正交的矩阵，保证其行列式不为0（正交矩阵行列式为1或-1）
        w_init = np.linalg.qr(w_init)[0].astype(np.float32)

        self.W = nn.Parameter(torch.from_numpy(w_init), requires_grad=True)
        # init.normal_(self.W, std=0.01)

        if dim == 1:
            self.equation = 'ij,bjn->bin'
            self.Ndim = 2
        elif dim == 2 or dim == -1:
            self.equation = 'ij,bnj->bni'
            self.Ndim = 1
        else:
            raise NotImplementedError(f"Unsupport dim {dim} for InvertibleConv1x1 Layer.")

    def forward(self, x: Tensor):
        z = torch.einsum(self.equation, self.W, x)      # 爱因斯坦求和约定，做矩阵乘法
        logdet = torch.slogdet(self.W)[1] * x.shape[self.Ndim]
        return z, logdet

    def inverse(self, z: Tensor):
        inv_W = torch.inverse(self.W)
        x = torch.einsum(self.equation, inv_W, z)
        logdet = -torch.slogdet(self.W)[1] * x.shape[self.Ndim]
        return x, logdet


if __name__ == '__main__':
    # w1 = np.random.randn(3, 3)
    # print(w1)
    # w2 = np.linalg.qr(w1)[0].astype(np.float32)
    # print(w2, np.linalg.det(w2))
    #
    # arr = np.zeros((3,), dtype=np.longlong)
    # print(arr)
    permutate1 = Permutation('inv1x1', 3, dim=2)
    x = torch.randn(2, 3, 3)
    c = torch.randn(2, 3, 4)
    y, log = permutate1(x, c)
    print(y.shape)
    print(log)
