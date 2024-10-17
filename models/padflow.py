from typing import List

import torch
import torch.nn as nn
from numpy.core.defchararray import str_len

from torch import Tensor
from torch.nn import functional as F

from pytorch3d.ops import knn_gather, knn_points

from models.flows.normalize import ActNorm
from models.flows.permutate import Permutation
from models.flows.coupling import AffineSpatialCouplingLayer, AffineCouplingLayer, AffineInjectorLayer
from utils.probs import GaussianDistribution


# 有条件/无条件 线性单元（Linear Unit），可用于仿射耦合/注入层计算scale和bias
class LinearA1D(nn.Module):

    def __init__(self, dim_in: int, dim_h: int, dim_out: int, dim_c=None):
        super(LinearA1D, self).__init__()
        linear_zero = nn.Linear(dim_h, dim_out, bias=True)
        linear_zero.weight.data.zero_()
        linear_zero.bias.data.zero_()

        in_channel = dim_in if dim_c is None else dim_in + dim_c

        self.layers = nn.Sequential(
            nn.Linear(in_channel, dim_h, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim_h, dim_h, bias=True),
            nn.LeakyReLU(inplace=True),
            linear_zero)

    def forward(self, h: Tensor, c: Tensor=None):
        if c is not None:
            # 融合辅助特征c，将输入特征和条件特征进行拼接
            h = torch.cat([h, c], dim=-1)
        h = self.layers(h)
        return h


# 简单的MLP对辅助特征做调整
class FeatMergeUnit(nn.Module):
    def __init__(self, idim: int, odim: int):
        super(FeatMergeUnit, self).__init__()
        self.conv1 = nn.Linear(idim, idim // 2, bias=True)
        self.conv2 = nn.Linear(idim // 2, odim, bias=False)

    def forward(self, x: Tensor):
        return self.conv2(F.relu(self.conv1(x)))


# 流模块
# 根据流模块在整个流中的奇偶位序交替地对通道划分
class FlowBlock(nn.Module):

    def __init__(self, idim, hdim, cdim, is_even):
        super(FlowBlock, self).__init__()

        # idim表示流模型中的通道个数，这里取3（点云坐标xyz，不做维度扩张），dim表示通道在shape中的位置。
        self.actnorm = ActNorm(idim, dim=2)
        self.permutate1 = Permutation('inv1x1', idim, dim=2)
        self.permutate2 = Permutation('reverse', idim, dim=2)

        # 在通道维度上进行划分split_dim=2
        if idim == 3:
            tdim = 1 if is_even else 2
            self.coupling1 = AffineSpatialCouplingLayer('additive', LinearA1D, split_dim=2, is_even=is_even, clamp=None,
                params={ 'dim_in' : tdim, 'dim_h': hdim, 'dim_out': idim - tdim, 'dim_c': cdim })
        else:
            self.coupling1 = AffineCouplingLayer('additive', LinearA1D, split_dim=2, clamp=None,
                params={ 'dim_in' : idim // 2, 'dim_h': hdim, 'dim_out': idim - idim // 2, 'dim_c': cdim })

        # 仿射注入层只使用辅助特征
        self.coupling2 = AffineInjectorLayer('affine', LinearA1D, split_dim=2, clamp=None,
                params={ 'dim_in': cdim, 'dim_h': hdim, 'dim_out': idim, 'dim_c': None })

    def forward(self, x: Tensor, c: Tensor=None):
        x, _log_det0 = self.actnorm(x)
        x, _log_det1 = self.permutate1(x, c)
        x, _log_det2 = self.coupling1(x, c)
        x, _log_det3 = self.permutate2(x, c)
        x, _log_det4 = self.coupling2(x, c)

        # _log_det2 = _log_det3 = 0
        return x, _log_det0 + _log_det1 + _log_det4

    def inverse(self, z: Tensor, c: Tensor=None):
        z, _ = self.coupling2.inverse(z, c)
        z, _ = self.permutate2.inverse(z, c)
        z, _ = self.coupling1.inverse(z, c)
        z, _ = self.permutate1.inverse(z, c)
        z, _ = self.actnorm.inverse(z)
        return z


# 特征提取单元，应用在点嵌入模块，编码得到辅助特征
class FeatureExtractUnit(nn.Module):

    def __init__(self, idim: int, odim: int, k: int, growth_width: int, is_dynamic: bool):
        super(FeatureExtractUnit, self).__init__()
        assert (odim % growth_width == 0)

        self.k = k
        self.is_dynamic = is_dynamic
        self.num_conv = (odim // growth_width)

        # 近邻翻3倍
        idim = idim * 3
        self.convs = nn.ModuleList()

        conv_first = nn.Sequential(*[
            nn.Conv2d(idim, growth_width, kernel_size=[1, 1]),
            nn.BatchNorm2d(growth_width),
            nn.LeakyReLU(0.05, inplace=True),
        ])
        self.convs.append(conv_first)

        for i in range(self.num_conv - 1):
            in_channel = growth_width * (i + 1) + idim

            conv = nn.Sequential(*[
                nn.Conv2d(in_channel, growth_width, kernel_size=[1, 1], bias=True),
                # nn.InstanceNorm1d(growth_width),
                nn.BatchNorm2d(growth_width),
                nn.LeakyReLU(0.05, inplace=True),
            ])
            self.convs.append(conv)

        self.conv_out = nn.Conv2d(growth_width * self.num_conv + idim, odim, kernel_size=[1, 1], bias=True)

    def derive_edge_feat(self, x: Tensor, knn_idx: Tensor or None):
        """
        x: [B, N, C]
        """
        if knn_idx is None and self.is_dynamic:
            # https://pytorch3d.readthedocs.io/en/latest/modules/ops.html#pytorch3d.ops.knn_points
            # 返回（距离，K近邻编号，K紧邻tensor），return_nn是否返回K近邻的tensor，return_sorted是否根据距离排序
            _, knn_idx, _ = knn_points(x, x, K=self.k, return_nn=False, return_sorted=False)  # [B, N, K]
        # 邻域k个点的特征
        knn_feat = knn_gather(x, knn_idx)  # [B, N, K, C]
        # x维度是[B, N, C]使用unsqueeze在-2位置插入维度，并复制k份，维度变成[B, N, K, C]
        x_tiled = torch.unsqueeze(x, dim=-2).expand_as(knn_feat)  # [B, N, K, C]

        return torch.cat([x_tiled, knn_feat, knn_feat - x_tiled], dim=-1)  # [B, N, K, C * 3]

    def forward(self, x: Tensor, knn_idx: Tensor or None, is_pooling=True):
        f = self.derive_edge_feat(x, knn_idx)  # [B, N, K, C * 3]
        f = f.permute(0, 3, 1, 2)  # [B, C * 3, N, K]

        for i in range(self.num_conv):
            _f = self.convs[i](f)
            f = torch.cat([f, _f], dim=1)       # Feature Grow Unit，输出和输入再拼接

        f = self.conv_out(f)  # [B, C, N, K]

        if is_pooling:
            f, _ = torch.max(f, dim=-1, keepdim=False)
            return torch.transpose(f, 1, 2)  # [B, N, C]
        else:
            return f


class PadFlow(nn.Module):

    def __init__(self, pc_channel: int):
        super(PadFlow, self).__init__()

        self.num_blocks = 6         # 标准化流过程中的流模块个数
        self.num_neighbors = 16     # knn算法的邻域点个数

        # 标准正态分布
        self.dist = GaussianDistribution(pc_channel, mu=0.0, vars=1.0, temperature=1.0)

        feat_channels = [pc_channel, 32, 64] + [128] * (self.num_blocks - 2)
        growth_widths = [8, 16] + [32] * (self.num_blocks - 2)
        cond_channels = [32, 64] + [128] * (self.num_blocks - 2)

        self.feat_convs = nn.ModuleList()
        for i in range(self.num_blocks):
            feat_conv = FeatureExtractUnit(feat_channels[i], feat_channels[i + 1], self.num_neighbors, growth_widths[i],
                                           is_dynamic=False)
            self.feat_convs.append(feat_conv)

        self.merge_convs = nn.ModuleList()
        for i in range(self.num_blocks):
            merge_unit = FeatMergeUnit(feat_channels[i + 1], cond_channels[i])
            self.merge_convs.append(merge_unit)

        self.flow_blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            step = FlowBlock(idim=pc_channel, hdim=64, cdim=cond_channels[i], is_even=(i % 2 == 0))
            self.flow_blocks.append(step)

    # 提取辅助特征
    def feat_extract(self, xyz: Tensor, knn_idx: Tensor):
        cs = []
        c = xyz

        for i in range(self.num_blocks):
            c = self.feat_convs[i](c, knn_idx=knn_idx)
            _c = self.merge_convs[i](c)
            cs.append(_c)
        return cs

    def f(self, xyz: Tensor, cs: List[Tensor]):
        (B, _, _), device = xyz.shape, xyz.device
        # 这里batch中的每一个点云都有对应的对数似然
        log_det_J = torch.zeros((B,), device=device)

        p = xyz

        for i in range(self.num_blocks):
            p, _log_det_J = self.flow_blocks[i](p, cs[i])
            if _log_det_J is not None:
                log_det_J += _log_det_J

        return p, log_det_J

    def g(self, z: Tensor, cs: Tensor, upratio: int):
        z = torch.flatten(z.transpose(2, 3), 1, 2)

        for i in reversed(range(self.num_blocks)):
            c = torch.repeat_interleave(cs[i], upratio, dim=1)
            z = self.flow_blocks[i].inverse(z, c)
        return z

    def set_to_initialized_state(self):
        for i in range(self.num_blocks):
            self.flow_blocks[i].actnorm.is_inited = True

    # pad-flow前向过程，
    def forward(self, xyz: Tensor, upratio=4):
        _, knn_idx, _ = knn_points(xyz, xyz, K=self.num_neighbors, return_nn=False, return_sorted=False)  # [B, N, K]

        p = xyz

        # 辅助特征
        cs = self.feat_extract(p, knn_idx)
        z, logp_x = self.log_prob(p, cs)

        # fz = self.interp(z, xyz, upratio)
        # x = self.g(fz, cs, upratio)
        x = z
        return x, logp_x

    # 计算对数似然
    def log_prob(self, xyz: Tensor, cs: List[Tensor]):
        x, log_det_J = self.f(xyz, cs)

        # standard gaussian probs
        logp_x = self.dist.standard_logp(x).to(x.device)
        logp_x = -torch.mean(logp_x + log_det_J)
        return x, logp_x

    def sample(self, sparse: Tensor, upratio=4):

        dense, _ = self(sparse, upratio)
        return dense

if __name__ == '__main__':
    # log_det_J = torch.zeros((3,), device='cuda')
    # xyz = torch.randn(2, 3, 3)
    # _, knn_idx, _ = knn_points(xyz, xyz, K=3, return_nn=False, return_sorted=False)
    # print(knn_idx)
    # print(log_det_J)
    i = 1
    while True:
        if i%3==0:
            break
        print(i)
        i+=1