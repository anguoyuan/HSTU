import abc
import logging
from typing import Dict, List, Tuple

import torch
from modeling.generic.initialization import truncated_normal
from modeling.generic.sequential.base_model import BaseModel
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import FeatConst
from torch import nn
import torch.nn.functional as F
from typing import List
from modeling.generic.utils.jagged_utils import dense_to_jagged, jagged_to_padded_dense

class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_cross_layers):
        super().__init__()
        self.num_cross_layers = num_cross_layers
        self.cross_w = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim))
            for _ in range(num_cross_layers)
        ])
        self.cross_bias = nn.ParameterList(
            [nn.Parameter(torch.randn(input_dim)) for _ in range(num_cross_layers)]
        )
    def forward(self, x):
        """
        x: [batch_size, seq_len, hidden_dim]
        对每个位置的 hidden_dim 做 Cross 操作
        """
        # lu
        x_0 = x  # 初始输入 [batch_size, cross_input_dim]
        x_l = x_0
        for i in range(self.num_cross_layers):
            # 计算 xlw = x_l · cross_w[i]（点积）
            xlw = torch.tensordot(x_l, self.cross_w[i], dims=1)  # [batch_size]
            # 计算 x_0 * xlw（广播乘法）
            xlw_expanded = xlw.unsqueeze(-1)  # [batch_size, 1]
            x_inter = x_0 * xlw_expanded  # [batch_size, cross_input_dim]
            x_l = x_inter + self.cross_bias[i] + x_l
        return x_l  # 输出形状 [batch_size, cross_input_dim]


class PPNetLayer(nn.Module):
    """PPNet 模块的 PyTorch 实现

    Args:
        feature_emb_dim (int): 特征嵌入维度
        gate_emb_dim (int): 门控嵌入维度
        mlp_layer_dims (list): MLP层部分的各个层输出维度
        gate_layer_dims (list): 各个gate层的中间维度
        use_bias (bool): 是否使用偏置，默认为 False
        act_func (str): 激活函数类型，默认为 "relu"
        is_act (bool): 是否对MLP使用激活函数，默认为 True
        batch_norm (bool): 是否使用 batch normalization，默认为 False
        dropout_rate (float): dropout 概率，默认为 0.1
        is_stop_grad (bool): 是否截断输入 feature_emb 的梯度，默认为 True
    """

    def __init__(self, feature_emb_dim: int, gate_emb_dim: int,
                 mlp_layer_dims: List[int], gate_layer_dims: List[int],
                 use_bias: bool = False, act_func: str = "relu",
                 is_act: bool = True, batch_norm: bool = False,
                 dropout_rate: float = 0.1, is_stop_grad: bool = True):
        super(PPNetLayer, self).__init__()

        # 验证输入维度
        if len(mlp_layer_dims) != len(gate_layer_dims):
            raise ValueError("mlp_layer_dims and gate_layer_dims must have the same length")

        self.feature_emb_dim = feature_emb_dim
        self.gate_emb_dim = gate_emb_dim
        self.use_bias = use_bias
        self.is_stop_grad = is_stop_grad
        self.mlp_layer_dims = mlp_layer_dims
        self.gate_layer_dims = gate_layer_dims
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.is_act = is_act
        self.act_func = act_func

        # 初始化激活函数
        self.activation = self._get_activation(act_func)

        # 初始化 dropout
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None

        # 计算门控输出维度
        self.gate_out_dims = [feature_emb_dim] + mlp_layer_dims[:-1]

        # 创建门控网络
        self.gates = nn.ModuleList()
        for i, (hidden_dim, out_dim) in enumerate(zip(gate_layer_dims, self.gate_out_dims)):
            gate_input_dim = feature_emb_dim + gate_emb_dim
            gate = nn.Sequential(
                nn.Linear(gate_input_dim, hidden_dim),
                self._get_activation(act_func),
                nn.Linear(hidden_dim, out_dim),
                nn.Sigmoid()  # 论文中 gate 输出使用 sigmoid
            )
            self.gates.append(gate)

        # 创建 MLP 层
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        # 第一层输入维度是特征嵌入维度
        input_dim = feature_emb_dim

        for i, output_dim in enumerate(mlp_layer_dims):
            layer = nn.Linear(input_dim, output_dim, bias=use_bias)
            self.layers.append(layer)

            # 添加 BatchNorm
            if batch_norm:
                # self.bn_layers.append(nn.BatchNorm1d(output_dim))
                self.bn_layers.append(nn.LayerNorm(output_dim))
            else:
                self.bn_layers.append(nn.Identity())  # 占位

            # 下一层的输入维度是当前层的输出维度
            input_dim = output_dim

    def _get_activation(self, act_func: str):
        """获取激活函数模块"""
        if act_func == "relu":
            return nn.ReLU()
        elif act_func == "sigmoid":
            return nn.Sigmoid()
        elif act_func == "tanh":
            return nn.Tanh()
        elif act_func == "leaky_relu":
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation function: {act_func}")

    def forward(self, inputs: List[torch.Tensor]):
        """
        Args:
            inputs (list): 包含两个张量 [feature_emb, gate_emb]
        Returns:
            torch.Tensor: 网络输出
        """
        feature_emb, gate_emb = inputs
        # 验证输入维度
        if feature_emb.size(-1) != self.feature_emb_dim:
            raise ValueError(f"Expected feature_emb dim {self.feature_emb_dim}, got {feature_emb.size(-1)}")
        if gate_emb.size(-1) != self.gate_emb_dim:
            raise ValueError(f"Expected gate_emb dim {self.gate_emb_dim}, got {gate_emb.size(-1)}")

        # 梯度截断
        if self.is_stop_grad:
            stop_feature_emb = feature_emb.detach()
        else:
            stop_feature_emb = feature_emb

        # 拼接 gate 输入
        gate_input = torch.cat([stop_feature_emb, gate_emb], dim=-1)

        hidden_out = feature_emb
        for i, (gate, layer, bn) in enumerate(zip(self.gates, self.layers, self.bn_layers)):
            # Gate 处理
            gate_out = 2 * gate(gate_input)  # 论文中 gate 输出乘以 2

            # 特征变换
            hidden_out = hidden_out * gate_out
            hidden_out = layer(hidden_out)
            hidden_out = bn(hidden_out)

            # 激活函数
            if self.is_act:
                hidden_out = self.activation(hidden_out)

            # Dropout
            if self.training and self.dropout is not None:
                hidden_out = self.dropout(hidden_out)

        return hidden_out


def get_activation(act_func: str):
        """获取激活函数模块"""
        if act_func == "relu":
            return nn.ReLU()
        elif act_func == "sigmoid":
            return nn.Sigmoid()
        elif act_func == "tanh":
            return nn.Tanh()
        elif act_func == "leaky_relu":
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation function: {act_func}")


class MLPLayer(nn.Module):

    def __init__(self, 
                 hidden_layers: List, 
                 act_func: str = 'relu', 
                 dropout_rate: float = 0.0, 
                 use_bn: bool = False):
        super().__init__()
        
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(hidden_layers[:-1], hidden_layers[1:])):
            if dropout_rate > 0.0:
                layers.append(torch.nn.Dropout(dropout_rate))
            layers.append(torch.nn.Linear(in_dim, out_dim))
            if use_bn:
                layers.append(torch.nn.BatchNorm1d(out_dim))
            if act_func and i != (len(hidden_layers) - 1): 
                layers.append(get_activation(act_func))
        print(f"gjx_log MLPLayer layers:{layers}")
        self.layers = torch.nn.Sequential(*layers)    

    def forward(self, input: torch.Tensor):
        return self.layers(input)
    

class MaskedBatchNorm1d(nn.Module):
    def __init__(self, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine 

        # 可训练参数
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1))
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        
        # 运行时的均值和方差（用于训练模式）
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_var', torch.ones(1))
    
    def forward(self, x:torch.Tensor, mask:torch.Tensor=None):
        '''
        Args:
            x: [B, C] or [N]
            mask: torch.Bool, [B, C] or [N]
        Returns:
            output: [B, C, 1] or [N, 1]
        '''
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.bool)
        else:
            mask = mask.bool()
        
        if self.training:
            # Compute batch statistics
            mean = (x * mask.float()).sum() / mask.float().sum() # []
            var = (((x - mean) ** 2) * mask.float()).sum() / mask.float().sum() # []

            # Update running stats
            with torch.no_grad():
                self.running_mean = (
                    (1 - self.momentum) * self.running_mean
                    + self.momentum * mean.view(*([1] * self.weight.ndim))
                )
                self.running_var = (
                    (1 - self.momentum) * self.running_var
                    + self.momentum * var.view(*([1] * self.bias.ndim))
                )
            mean = mean.view(*([1] * x.ndim))
            var = var.view(*([1] * x.ndim))
        else:
            # Use running stats during inference
            mean = self.running_mean.view(*([1] * x.ndim))
            var = self.running_var.view(*([1] * x.ndim))

        # Normalize
        output = (x - mean) / torch.sqrt(var + self.eps)

        # Apply gamma and beta with broadcast
        if self.affine:
            output = self.weight.view(*([1] * x.ndim)) * output + \
                self.bias.view(*([1] * x.ndim))
            
        output[~mask] = 0.0
        output = output.unsqueeze(-1)
        return output
