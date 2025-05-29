import torch
import torch_geometric
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F
import torch.nn as nn


class WGCN(MessagePassing):

    def __init__(self, in_channels, out_channels, heads=, negative_slope=, dropout=, weighted=True):
        super(WGCN, self).__init__(aggr='add', node_dim=)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.weighted = weighted

        # 初始化权重参数
        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))

        if weighted:
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels + 1))
        else:
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        # 增加 feature_weight 参数，用于邻居节点特征的加权
        self.feature_weight = Parameter(torch.Tensor(1, heads, out_channels))




        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        glorot(self.feature_weight)

    def forward(self, x, edge_index, edge_attr=None):
        # 输入的残差连接
        x_residual = x.clone()

        # 线性变换
        x = torch.matmul(x, self.weight)

        # 传播消息
        out = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

        # 多头注意力平均
        out = out.mean(dim=1)

        # 添加残差连接
        out += x_residual


        return out

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr=None):
        x_j = x_j.view(-1, self.heads, self.out_channels)
        x_i = x_i.view(-1, self.heads, self.out_channels)

        if self.weighted and edge_attr is not None:
            alpha = (torch.cat([x_i, x_j, edge_attr.view(-1, 1).repeat(1, self.heads).view(-1, self.heads, 1)],
                               dim=-1) * self.att).sum(dim=-1)
        else:
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = torch_geometric.utils.softmax(src=alpha, index=edge_index_i, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 使用 feature_weight 参数对邻居特征进行加权
        result = x_j * alpha.view(-1, self.heads, 1) * self.feature_weight

        return result