import abc
import math
from typing import Dict, Tuple, List, Optional
import logging
import torch
import torch.nn as nn 
import torch_npu
import torch.nn.functional as F
from modeling.generic.sequential.rab_modules import RABModule
from modeling.generic.sequential.utils import handle_padded_qk
from modeling.generic.sequential.base_model import BaseModel
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const
from modeling import HAS_ATTN_FUSION_OPS, ENABLE_JAGGED_OPS
from modeling.generic.utils.jagged_utils import dense_to_jagged, jagged_to_padded_dense
from modeling.generic.utils.hstu_dense_utils import hstu_dense
from modeling.generic.utils.hstu_fuxi_utils import hstu_fuxi
from torch.autograd.profiler import record_function


TransformerCacheState = Const.TransformerCacheState


class TransformerCache:
    def __init__(self, n: int = 0):
        self.cached_v = torch.tensor([])
        self.cached_q = torch.tensor([])
        self.cached_k = torch.tensor([])
        self.cached_outputs = torch.tensor([])
        self.n = 0

    def append(self, cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        向缓存中增加新的元素
        """
        v, q, k, outputs = cache
        is_same_number = all([t.shape[0] == self.n for t in (v, q, k, outputs)])
        if self.n == 0 or is_same_number:
            self.cached_v = torch.cat((self.cached_v, v), dim=0)
            self.cached_q = torch.cat((self.cached_q, q), dim=0)
            self.cached_k = torch.cat((self.cached_k, k), dim=0)
            self.cached_outputs = torch.cat((self.cached_outputs, outputs), dim=0)
            self.n += 1
        else:
            raise ValueError("New elements must have the same number of caches as the current cache size.")

    def select(self, index: int = 0):
        """
        根据索引取出特定的缓存元素
        """
        if index < 0 or index >= self.n:
            raise IndexError("Index out of range.")

        return self.cached_v[index], self.cached_q[index], self.cached_k[index], self.cached_outputs[index]


class FeedForward(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int,  dropout: float):
        super().__init__()
        # self.w1 = torch.nn.Linear(dim, hidden_dim, bias=False)
        # self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False)
        # self.w3 = torch.nn.Linear(dim, hidden_dim, bias=False)
    #     self.dropout = torch.nn.Dropout(dropout)
        

    # def forward(self, x):
    #     return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

        self.w1 = torch.nn.Linear(dim, hidden_dim * 2, bias=False)
        self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(torch_npu.npu_swiglu(self.w1(x), dim=-1))) 
        
class RMSNorm_npu(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class GLUFFN(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int, 
        multiple_of: int = 2,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1: Linear transformation for the first layer.
            w2: Linear transformation for the second layer.
            w3: Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # self.w1 = torch.nn.Linear(input_dim, hidden_dim, bias=False)
        # self.w2 = torch.nn.Linear(hidden_dim, output_dim, bias=False)
        # self.w3 = torch.nn.Linear(input_dim, hidden_dim, bias=False)

    # def forward(self, x):
    #     return self.w2(F.silu(self.w1(x)) * self.w3(x))
        self.w1 = torch.nn.Linear(input_dim, hidden_dim * 2, bias=False)
        self.w2 = torch.nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        return self.w2(torch_npu.npu_swiglu(self.w1(x), dim=-1))
    

class ScaledDotProductAttention(torch.nn.Module):
    """ scaled dot product cross attention 
        Ref: https://zhuanlan.zhihu.com/p/47812375
    """
    def __init__(self, embedding_dim, num_heads, attention_dim, dropout_rate=0.):
        super(ScaledDotProductAttention, self).__init__()
        self._embedding_dim = embedding_dim 
        self._num_heads = num_heads 
        self._attention_dim = attention_dim 

        # q, k, v linear transformation 
        self.q_linear = torch.nn.Linear(self._embedding_dim, self._attention_dim * self._num_heads, bias=False)
        self.k_linear = torch.nn.Linear(self._embedding_dim, self._attention_dim * self._num_heads, bias=False)
        self.v_linear = torch.nn.Linear(self._embedding_dim, self._attention_dim * self._num_heads, bias=False)
        self.o_linear = torch.nn.Linear(self._attention_dim * self._num_heads, self._embedding_dim, bias=False)

        self.dropout = torch.nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None, rel_attn_mask=None):
        """
        args:
            Q: [B, C, D_in]
            K: [B, L, D_in]
            V: [B, L, D_in]
            mask: [B, C, L], False for masked positions
            rel_attn_mask: [B, H, Q, L]  
        returns: 
            output: [B, C, D_in]
            attention: [B, H, C, L]
        """
        
        B, C, D = Q.shape
        _, L, _ = K.shape

        # transformation 
        Q = self.q_linear(Q).view(B, C, self._num_heads, self._attention_dim).transpose(1, 2) # [B, H, C, D_a]
        K = self.k_linear(K).view(B, L, self._num_heads, self._attention_dim).transpose(1, 2) # [B, H, L, D_a]
        V = self.k_linear(V).view(B, L, self._num_heads, self._attention_dim).transpose(1, 2) # [B, H, L, D_a]

        # multiply and scale
        scores = torch.matmul(Q, K.transpose(-1, -2)) # [B, H, C, L]
        scaling = float(self._attention_dim) ** -0.5
        scores = scores * scaling
        
        # relative attention bias 
        if rel_attn_mask is not None:
            scores = scores + rel_attn_mask

        # mask
        if mask is not None:
            mask = torch.unsqueeze(mask, dim=1) # [B, 1, C, L]
            scores = scores.masked_fill_(mask.float() == 0, -1.e9) # fill -inf if mask=0
        
        # normalization
        attention = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)

        # output 
        output = torch.matmul(attention, V) # [B, H, C, D_a]
        output = output.transpose(1, 2).reshape(B, C, self._num_heads * self._attention_dim) # [B, C, H * D]
        output = self.o_linear(output) # [B, C, D_in]

        return output, attention
    

class Transformer(BaseModel):
    """
    基础的 Sequential Transduction Unit, STU 用于处理序列数据.
    """
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        model_conf = common_hp.get("model_conf")
        sequential_module_config = model_cfg[Const.HP]
        self._embedding_dim: int = model_conf.get("item_embedding_dim", 128)
        self._linear_dim: int = sequential_module_config.get("dv", 32)
        self._attention_dim: int = sequential_module_config.get("dqk", 32)
        self._num_heads: int = sequential_module_config.get("num_heads", 4)
        self._linear_config: str = sequential_module_config.get("linear_config", "uvqk")
        self._linear_activation: str = sequential_module_config.get("linear_activation", "silu")
        self._dropout_ratio: float = model_conf.get("linear_dropout_rate", 0.3)
        self._attn_dropout_ratio: float = model_conf.get("attn_dropout_rate", 0.0)
        self._normalization: str = model_conf.get("normalization", "rel_bias")
        self._max_sequence_length: int = model_conf.get("max_sequence_length", 512)
        self._rel_attn_bias: RABModule = self.init_sub_model("RABModule") if "RABModule" in model_cfg[Const.SUB_MODELS] \
                                                        else None
        self._eps: float = Const.EPS

        if self._linear_config == "uvqk":
            self._uvqk = torch.nn.Parameter(
                torch.empty((self._embedding_dim, self._linear_dim * 2 * self._num_heads +
                             self._attention_dim * self._num_heads * 2)).normal_(mean=0, std=0.02), )
        else:
            raise ValueError("Unknown linear_config %s", self._linear_config)

        self._o = torch.nn.Linear(in_features=self._linear_dim * self._num_heads, out_features=self._embedding_dim)
        torch.nn.init.xavier_uniform_(self._o.weight)

        self.layer_norm_input = RMSNorm_npu(self._embedding_dim, eps=self._eps)
        self.layer_norm_attn_output = RMSNorm_npu(self._linear_dim * self._num_heads, eps=self._eps)

        qk_attn_denominator = sequential_module_config.get("qk_attn_denominator", "emb_dim")
        if qk_attn_denominator == "emb_dim":
            self.qk_attn_denominator_value = 1 / self._embedding_dim
        elif qk_attn_denominator == "sqrt_d":
            self.qk_attn_denominator_value = 1 / math.sqrt(self._embedding_dim)
        elif qk_attn_denominator == "max_seq_len":
            self.qk_attn_denominator_value = 1 / (self._max_sequence_length * 2 + 2)
        else:
            raise ValueError("Unknown string %s", qk_attn_denominator)

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm_input(x)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm_attn_output(x)

    """
    线性变换层用于从原始的 x 输出 q, k, v.
    """
    def _linear_transform(self, normed_x: torch.Tensor) -> torch.Tensor:
        if self._linear_config == "uvqk":
            batched_mm_output = torch.matmul(normed_x, self._uvqk)
            if self._linear_activation == "silu":
                batched_mm_output = F.silu(batched_mm_output)
            elif self._linear_activation == "none":
                batched_mm_output = batched_mm_output
            # u 特征交互, qkv transformer
            u, v, q, k = torch.split(
                batched_mm_output,
                [self._linear_dim * self._num_heads, self._linear_dim * self._num_heads,
                 self._attention_dim * self._num_heads, self._attention_dim * self._num_heads],
                dim=-1,
            )
            return u, v, q, k
        else:
            raise ValueError("Unknown linear_config %s", self._linear_config)

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def forward(
            self,
            x: torch.Tensor,
            x_offsets: torch.Tensor,
            all_timestamps: torch.Tensor,
            invalid_attn_mask: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            layer_num: int,
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            cache: TransformerCacheState = (torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])),
            return_cache_states: bool = False,
            time_bias: torch.Tensor = None
    ) -> Tuple[torch.Tensor, TransformerCacheState, torch.Tensor, torch.Tensor]:
        pass        


@ModelRegistry.register(opt_subs={"RABModule"})
class HSTU(Transformer):
    """
    HSTU模型用于处理序列数据.
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):

        """
        继承父类Transformer的参数
        """
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        feat_conf = common_hp.get("feature_conf")
        fuse_ia = feat_conf.get("fuse_ia", False)
        self.token_per_item = 2 if not fuse_ia else 1

        model_hp = model_cfg.get(Const.HP, {})
        print(f'HSTU model_hp: {model_hp}')
        self.ffn_type = model_hp.get('ffn_type', None) # "ffn" or "glu_ffn" or None 
        self.ffn_expand = model_hp.get('ffn_expand', 6)
        if self.ffn_type is not None: 
            self.norm_ffn = RMSNorm_npu(self._embedding_dim, eps=self._eps)
            if self.ffn_type == 'ffn':
                self.feed_forward = FeedForward(
                    dim=self._embedding_dim,
                    hidden_dim=int(self._embedding_dim * self.ffn_expand),
                    dropout=self._dropout_ratio,
                )
            elif self.ffn_type == 'glu_ffn':
                self.feed_forward = GLUFFN(
                    input_dim=self._embedding_dim, 
                    hidden_dim=self._embedding_dim, 
                    output_dim=self._embedding_dim, 
                    ffn_dim_multiplier=self.ffn_expand
                )
            else:
                raise ValueError('ffn_type must be chosen in ["ffn", "glu_ffn"]')

    def forward(
            self,
            x: torch.Tensor,
            x_offsets: torch.Tensor,
            fusion_enabled: bool,
            jagged_enabled: bool,
            all_timestamps: torch.Tensor,
            attn_mask: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            layer_num: int,
            cache: TransformerCacheState = (torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])),
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            return_cache_states: bool = False,
            time_bias: torch.Tensor = torch.tensor([])
    ) -> Tuple[torch.Tensor, TransformerCacheState, torch.Tensor]:
        """
        前向传播方法, 处理输入序列并生成输出序列.

        :param x: 输入序列的特征, 形状为(\sum_i N_i, D).
        :param x_offsets: 输入序列的偏移量, 形状为(B + 1), 表示每个序列的起始位置.
        :param all_timestamps: 可选参数, 时间戳序列, 形状为(B, N).
        :param attn_mask: 无效的注意力掩码, 形状为(B, N, N), 每个元素为0或1.
        :param past_lengths: 过去序列的长度, 形状为(B,).
        :param num_rerank: 需要推理部分的长度.
        :param layer_num: 当前层的编号.
        :param delta_x_offsets: 可选参数, 形状为((B,), (B,))的偏移量, 对于元组中的第一个元素, 
            每个元素在[0,x_offsets[-1])中. 对于元组中的第2个元素, 每个元素在[0,N)中.
        :param cache: 可选参数, 缓存状态, 用于存储中间结果(v, padded_q, padded_k, output).
        :param return_cache_states: 是否返回缓存状态.
        :return: 处理后的输出序列, 形状为(\sum_i N_i, D).
        """
        # n 代表整个需要推理的序列长度
        n: int = attn_mask.shape[-1]
        cached_v = torch.zeros_like(x, device=x.device)
        cached_q = torch.zeros_like(x, device=x.device)
        cached_k = torch.zeros_like(x, device=x.device)
        cached_outputs = torch.zeros_like(x, device=x.device)
        if delta_x_offsets[0].shape[0] > 0:
            # In this case, for all the following code, x, u, v, q, k become restricted to
            # 维度 [delta_x_offsets[0], :].
            if cache[0][0].shape[0] == 0:
                raise ValueError("cache must be provided when delta_x_offsets is not None")
            x = x[delta_x_offsets[0], :]
            cached_v, cached_q, cached_k, cached_outputs = cache

        normed_x = self._norm_input(x)

        if self._linear_config == "uvqk":
            u, v, q, k = self._linear_transform(normed_x)
        else:
            raise ValueError("Unknown linear_config %s", self._linear_config)

        if delta_x_offsets[0].shape[0] > 0:
            v = cached_v.index_copy_(dim=0, index=delta_x_offsets[0], source=v)

        bs: int = x_offsets.shape[0] - 1
        if self._normalization == "rel_bias":
            if delta_x_offsets[0].shape[0] > 0:
                k, q = handle_padded_qk(bs, cached_k, cached_q, delta_x_offsets, k, n, q)

            rel_attention_mask = None
            if all_timestamps is not None and self._rel_attn_bias is not None:
                # Relative Attention Bias --> attention bias
                # reshape qk_attn:  batch x num_head x 2n x 2n  --> batch x num_head x n x 2 x n x 2
                rel_attention_mask, time_bias = self._rel_attn_bias(
                         all_timestamps, past_lengths, num_rerank,
                         layer_num, time_bias)
                # 形如 [bs, _num_heads, (n-1), (n-1)]
                rel_attention_mask = rel_attention_mask.unsqueeze(1).repeat(1, self._num_heads, 1, self.token_per_item)
                seq_tokens = n // self.token_per_item - 1
                rel_attention_mask = rel_attention_mask.view(
                    bs, self._num_heads, seq_tokens, 1, seq_tokens
                )
                rel_attention_mask = rel_attention_mask.repeat(1, 1, 1, 1, self.token_per_item)
                rel_attention_mask = rel_attention_mask.transpose(-1, -2).reshape(bs, self._num_heads, n - 1, n - 1)
                # 形如 [bs, _num_heads, n, n]，补上user
                rel_attention_mask = torch.nn.functional.pad(rel_attention_mask, (1, 0, 1, 0), 'constant', 0.0)

            if fusion_enabled:
                # mask = torch.tril(torch.ones((bs, self._num_heads, n, n), dtype=torch.int64, device=q.device))
                mask_type = 3  # custom
                if jagged_enabled:
                    qk_shape = (-1, self._num_heads, self._attention_dim)
                    v_shape = (-1, self._num_heads, self._linear_dim)
                    layout = "jagged"
                    seq_offset = x_offsets.tolist()
                    out_shape = (-1, self._num_heads * self._linear_dim)
                else:
                    qk_shape = (bs, n, self._num_heads, self._attention_dim)
                    v_shape = (bs, n, self._num_heads, self._linear_dim)
                    layout = "normal"
                    seq_offset = None
                    out_shape = (bs, n, self._num_heads * self._linear_dim)

                # input:(q, k, v, mask, rab, mask_type, max_seq_len, silu_scale, layout, seq_offset=None)
                # mask_type: 0 tril, 1 triu, 2 none, 3 custom. layout: "normal" padding, "jagged" non-padding
                attn_output = hstu_dense(
                    q.view(qk_shape), k.view(qk_shape), v.view(v_shape), attn_mask, rel_attention_mask, mask_type,
                    n, self.qk_attn_denominator_value, layout, seq_offset
                ).reshape(out_shape)
            else:
                # 形如 [B, H, N, N]
                qk_attn = torch.einsum(
                    "bnhd,bmhd->bhnm",
                    q.view(bs, n, self._num_heads, self._attention_dim),
                    k.view(bs, n, self._num_heads, self._attention_dim),
                )

                if rel_attention_mask is not None:
                    qk_attn = qk_attn + rel_attention_mask
                qk_attn = F.silu(qk_attn) * self.qk_attn_denominator_value
                attn_mask = attn_mask.to(qk_attn.device)
                # 形如 [B, 1, N, N]
                attn_mask = attn_mask.unsqueeze(1)
                qk_attn = qk_attn * attn_mask
                attn_output = torch.einsum(
                    "bhnm,bmhd->bnhd",
                    qk_attn,
                    v.view(bs, n, self._num_heads, self._linear_dim)
                ).reshape(bs, n, self._num_heads * self._linear_dim)
        else:
            raise ValueError("Unknown normalization method %s", self._normalization)

        attn_output = attn_output if delta_x_offsets[0].shape[0] == 0 else attn_output[delta_x_offsets[0], :]
        o_input = u * self._norm_attn_output(attn_output)
        # x --> u k q v
        new_outputs = self._o(
            F.dropout(
                o_input,
                p=self._dropout_ratio,
                training=self.training,
            )
        ) + x

        ## HSTU引入FFN层
        if self.ffn_type:
            # norm + ffn + add
            ffn_input = self.norm_ffn(new_outputs)
            new_outputs = self.feed_forward(ffn_input) + new_outputs

        if delta_x_offsets[0].shape[0] > 0:
            new_outputs = cached_outputs.index_copy_(dim=0, index=delta_x_offsets[0], source=new_outputs)

        if return_cache_states and delta_x_offsets[0].shape[0] == 0:
            v = v.contiguous()

        return new_outputs, (v, q, k, new_outputs), time_bias

@ModelRegistry.register(req_hp=True, opt_subs={"RABModule"})
class FUXI(Transformer):
    """
    HSTU模型用于处理序列数据.
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):

        """
        继承父类Transformer的参数
        """
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        ffn_expand = model_cfg[Const.HP].get("ffn_expand")
        ## 判断attention的linear head的数目
        ## 如果attention和位置，时间bias均存在，则linear head的数目为3 + 1 = 4
        if self._normalization == "rel_bias" and self._rel_attn_bias is not None:
            self.linear_number = 4
        ## 如果attention存在，位置，时间bias不存在，则linear head的数目为1 + 1 = 2
        elif self._normalization == "rel_bias" and self._rel_attn_bias is None:
            self.linear_number = 2
        ## 如果attention不存在，位置，时间bias存在，则linear head的数目为2 + 1 = 3            
        elif self._normalization == "att_free_bias" and self._rel_attn_bias is not None:
            self.linear_number = 3
        ## 其他情况下模型报错
        else:
            raise ValueError("error, check the configuration.")

        if self._linear_config == "uvqk":
            self._uvqk = torch.nn.Parameter(
                torch.empty((self._embedding_dim, self.linear_number * self._linear_dim * self._num_heads +
                            self._attention_dim * self._num_heads * 2)).normal_(mean=0, std=0.02), )
        else:
            raise ValueError("Unknown linear_config %s", self._linear_config)  
        
        self._o = torch.nn.Linear(
            in_features=(self.linear_number - 1) * self._linear_dim * self._num_heads,
            out_features=self._embedding_dim
        )

        torch.nn.init.xavier_uniform_(self._o.weight)

        self.layer_norm_attn_output = RMSNorm_npu(
            (self.linear_number - 1) * self._linear_dim * self._num_heads,
            eps=self._eps
        )

        self.layer_norm_ffn = RMSNorm_npu(self._embedding_dim, eps=self._eps)

        self.ffn_expand = ffn_expand
        self.feed_forward = FeedForward(
            dim=self._embedding_dim,
            hidden_dim=int(self._embedding_dim * ffn_expand),
            dropout=self._dropout_ratio,
        )

        feat_conf = common_hp.get("feature_conf")
        fuse_ia = feat_conf.get("fuse_ia", False)
        self.token_per_item = 2 if not fuse_ia else 1

    def _norm_ffn(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm_ffn(x)

    def _linear_transform(self, normed_x: torch.Tensor) -> torch.Tensor:
        if self._linear_config == "uvqk":
            batched_mm_output = torch.matmul(normed_x, self._uvqk)
            if self._linear_activation == "silu":
                batched_mm_output = F.silu(batched_mm_output)
            elif self._linear_activation == "none":
                batched_mm_output = batched_mm_output
            # u 特征交互, qkv transformer
            u, v, q, k = torch.split(
                batched_mm_output,
                [(self.linear_number - 1) * self._linear_dim * self._num_heads, self._linear_dim * self._num_heads,
                 self._attention_dim * self._num_heads, self._attention_dim * self._num_heads],
                dim=-1,
            )
            return u, v, q, k
        else:
            raise ValueError("Unknown linear_config %s", self._linear_config)
        
    def forward(
            self,
            x: torch.Tensor,
            x_offsets: torch.Tensor,
            fusion_enabled: bool,
            jagged_enabled: bool,
            all_timestamps: torch.Tensor,
            attn_mask: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            layer_num: int,
            cache: TransformerCacheState = (torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])),
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            return_cache_states: bool = False,
            time_bias: torch.Tensor = torch.Tensor([])
    ) -> Tuple[torch.Tensor, TransformerCacheState, torch.Tensor]:
        """
        前向传播方法, 处理输入序列并生成输出序列.

        :param x: 输入序列的特征, 形状为(\sum_i N_i, D).
        :param x_offsets: 输入序列的偏移量, 形状为(B + 1), 表示每个序列的起始位置.
        :param all_timestamps: 可选参数, 时间戳序列, 形状为(B, N).
        :param attn_mask: 无效的注意力掩码, 形状为(B, N, N), 每个元素为0或1.
        :param past_lengths: 过去序列的长度, 形状为(B,).
        :param num_rerank: 需要推理部分的长度.
        :param layer_num: 当前层的编号.
        :param delta_x_offsets: 可选参数, 形状为((B,), (B,))的偏移量, 对于元组中的第一个元素, 
            每个元素在[0,x_offsets[-1])中. 对于元组中的第2个元素, 每个元素在[0,N)中.
        :param cache: 可选参数, 缓存状态, 用于存储中间结果(v, padded_q, padded_k, output).
        :param return_cache_states: 是否返回缓存状态.
        :return: 处理后的输出序列, 形状为(\sum_i N_i, D).
        """
        # n 代表整个需要推理的序列长度
        n: int = attn_mask.shape[-1]
        cached_v = torch.zeros_like(x, device=x.device)
        cached_q = torch.zeros_like(x, device=x.device)
        cached_k = torch.zeros_like(x, device=x.device)
        cached_outputs = torch.zeros_like(x, device=x.device)
        if delta_x_offsets[0].shape[0] > 0:
            # In this case, for all the following code, x, u, v, q, k become restricted to
            # 维度 [delta_x_offsets[0], :].
            if cache[0][0].shape[0] == 0:
                raise ValueError("cache must be provided when delta_x_offsets is not None")
            x = x[delta_x_offsets[0], :]
            cached_v, cached_q, cached_k, cached_outputs = cache

        normed_x = self._norm_input(x)

        if self._linear_config == "uvqk":
            u, v, q, k = self._linear_transform(normed_x)
        else:
            raise ValueError("Unknown linear_config %s", self._linear_config)

        if delta_x_offsets[0].shape[0] > 0:
            v = cached_v.index_copy_(dim=0, index=delta_x_offsets[0], source=v)

        bs: int = x_offsets.shape[0] - 1

        # fuxi-alpha，保留q * k的 attention 计算矩阵
        if self._normalization == "rel_bias":
            if delta_x_offsets[0].shape[0] > 0:
                k, q = handle_padded_qk(bs, cached_k, cached_q, delta_x_offsets, k, n, q)

            if fusion_enabled:
                mask_type = 3  # custom
                if jagged_enabled:
                    qk_shape = (-1, self._num_heads, self._attention_dim)
                    v_shape = (-1, self._num_heads, self._linear_dim)
                    layout = "jagged"
                    seq_offset = x_offsets.tolist()
                    out_shape = (-1, self._num_heads * self._linear_dim)
                else:
                    raise ValueError("It is impossible to set enable_jagged_ops to False while using fusion_ops of hstu_fuxi")
                    qk_shape = (bs, n, self._num_heads, self._attention_dim)
                    v_shape = (bs, n, self._num_heads, self._linear_dim)
                    layout = "normal"
                    seq_offset = None
                    out_shape = (bs, n, self._num_heads * self._linear_dim)

                # input:(q, k, v, mask, rab, mask_type, max_seq_len, silu_scale, layout, seq_offset=None)
                # mask_type: 0 tril, 1 triu, 2 none, 3 custom. layout: "normal" padding, "jagged" non-padding
                attn_output = hstu_fuxi(
                        q.view(qk_shape), k.view(qk_shape), v.view(v_shape), None, None, attn_mask, 
                        mask_type, n, self.qk_attn_denominator_value, layout, seq_offset)

            else:
                # 形如 [B, H, N, N]
                qk_attn = torch.einsum(
                    "bnhd,bmhd->bhnm",
                    q.view(bs, n, self._num_heads, self._attention_dim),
                    k.view(bs, n, self._num_heads, self._attention_dim),
                )

                qk_attn = F.silu(qk_attn) * self.qk_attn_denominator_value
                attn_mask = attn_mask.to(qk_attn.device)
                # 形如 [B, 1, N, N]
                attn_mask = attn_mask.unsqueeze(1)
                qk_attn = qk_attn * attn_mask
                attn_output = torch.einsum(
                    "bhnm,bmhd->bnhd",
                    qk_attn,
                    v.view(bs, n, self._num_heads, self._linear_dim)
                ).reshape(bs, n, self._num_heads * self._linear_dim)
        
        # fuxi-beta，去掉attention计算矩阵
        elif self._normalization == "att_free_bias":
            attn_mask = attn_mask.to(q.device)
            # 形如 [B, 1, N, N]
            attn_mask = attn_mask.unsqueeze(1)
        
        else:
            raise ValueError("Unknown normalization method %s", self._normalization)          

        if all_timestamps is not None and self._rel_attn_bias is not None:
            # Relative Attention Bias --> attention bias
            # reshape qk_attn:  batch x num_head x 2n x 2n  --> batch x num_head x n x 2 x n x 2

            if torch.onnx.is_in_onnx_export() or num_rerank > 0:
                # 分档推理
                # rel_attention_mask 形如 [bs, (n-2)//2, (n-2)//2]
                rel_attention_mask, time_bias = self._rel_attn_bias(
                    all_timestamps, past_lengths, num_rerank,
                    layer_num, time_bias, (n - num_rerank - 1) // self.token_per_item)
            else:
                rel_attention_mask, time_bias = self._rel_attn_bias(
                    all_timestamps, past_lengths, num_rerank,
                    layer_num, time_bias)
            # 对于fuxi模型，rab_aggregate_method应该为concat，attention_mask的中间维度应该为2*，判断是否配置错误
            if len(rel_attention_mask.shape) != 4:
                logging.error("the rab_aggregate_method should be configured as concat.")

            # 形如 [bs, 2, (n-1), (n-1)]
            rel_attention_mask = rel_attention_mask.repeat(1, 1, 1, self.token_per_item)  # b, 2, (n-2)/2, n-2
            seq_tokens = n // self.token_per_item - 1
            rel_attention_mask = rel_attention_mask.view(
                bs, 2, seq_tokens, 2, seq_tokens
            ).repeat(1, 1, 1, 1, self.token_per_item)  # b, 2, (n-2)/2, 2, n-2
            rel_attention_mask = rel_attention_mask.transpose(-1, -2).reshape(bs, 2, n - 1, n - 1)
            rel_attention_mask = torch.nn.functional.pad(rel_attention_mask, (1, 0, 1, 0), 'constant', 0.0)
            rel_attention_mask = rel_attention_mask * attn_mask
            rel_attn_output = torch.einsum(
                "bhnm,bmd->bnhd",
                rel_attention_mask,
                v.view(bs, n, self._num_heads * self._linear_dim)
            ).reshape(bs, n, 2 * self._num_heads * self._linear_dim) 

        if self._normalization == "rel_bias" and self._rel_attn_bias is not None:
            attn_output = torch.cat([attn_output, rel_attn_output], 2)
        elif self._normalization == "rel_bias" and self._rel_attn_bias is None:
            attn_output = attn_output
        elif self._normalization == "att_free_bias" and self._rel_attn_bias is not None:
            attn_output = rel_attn_output
        else:
            raise ValueError("error, check the configuration.")
        
        attn_output = attn_output if delta_x_offsets[0].shape[0] == 0 else attn_output[delta_x_offsets[0], :]
        o_input = u * self._norm_attn_output(attn_output)
        # x --> u k q v
        new_outputs = self._o(
            F.dropout(
                o_input,
                p=self._dropout_ratio,
                training=self.training,
            )
        ) + x

        ## fuxi-alpha引入FFN层
        ffn_input = self._norm_ffn(new_outputs)
        ffn_output = self.feed_forward.forward(ffn_input) + new_outputs

        if delta_x_offsets[0].shape[0] > 0:
            ffn_output = cached_outputs.index_copy_(dim=0, index=delta_x_offsets[0], source=ffn_output)

        if return_cache_states and delta_x_offsets[0].shape[0] == 0:
            v = v.contiguous()

        return ffn_output, (v, q, k, ffn_output), time_bias

@ModelRegistry.register(req_hp=True, req_subs={"Transformer"})
class SequentialModule(BaseModel):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        sequential_module_config = model_cfg[Const.HP]
        model_conf = common_hp["model_conf"]
        self.num_blocks = sequential_module_config.get("num_blocks", 8)
        self.num_heads = model_cfg[Const.SUB_MODELS]["Transformer"]["hp"].get("num_heads", 4)
        self.enable_fusion_ops = model_cfg[Const.SUB_MODELS]["Transformer"]["hp"].get("enable_fusion_ops", False)
        self.enable_jagged_ops = model_conf.get("enable_jagged_ops", False)
        self._transformer = TransformerInner(
                            modules=[self.init_sub_model("Transformer") for _ in range(self.num_blocks)],
                            num_heads=self.num_heads,
                            enable_fusion_ops=self.enable_fusion_ops,
                            enable_jagged_ops=self.enable_jagged_ops
        )

    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: torch.Tensor,
        attn_mask: torch.Tensor,
        past_lengths: torch.Tensor,
        num_rerank: int,
        cache: Optional[List[TransformerCacheState]] = None,
        delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
        return_cache_states: bool = False,
    ) -> Tuple[torch.Tensor, List[TransformerCacheState]]:

        return self._transformer(
            x=x,
            x_offsets=x_offsets,
            all_timestamps=all_timestamps,
            attn_mask=attn_mask,
            past_lengths=past_lengths,
            num_rerank=num_rerank,
            cache=cache,
            delta_x_offsets=delta_x_offsets,
            return_cache_states=return_cache_states,
        )

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass    


class TransformerInner(torch.nn.Module):
    """
    TransformerInner类, 用于封装一系列Transformer模块, 如LlaMa, HSTU, Fuxi等, 实现分层序列建模.
    该类负责管理多个Transformer模块, 并提供前向传播接口.
    """

    def __init__(
            self,
            modules: List[Transformer],
            num_heads: int,
            enable_fusion_ops: bool,
            enable_jagged_ops: bool
    ) -> None:
        super().__init__()
        self._attention_layers: torch.nn.ModuleList = torch.nn.ModuleList(modules=modules)
        self._num_heads: int = num_heads
        self.enable_fusion_ops = enable_fusion_ops and HAS_ATTN_FUSION_OPS and self.training
        self.enable_jagged_ops = enable_jagged_ops and self.training
        if not self.enable_fusion_ops and self.enable_jagged_ops:
            raise ValueError("It is impossible to set enable_jagged_ops to True while setting enable_fusion_ops to False")
        elif self.enable_fusion_ops and self.enable_jagged_ops:
            logging.info("Enable attn fusion ops with jagged mode.")
        elif self.enable_fusion_ops and not self.enable_jagged_ops:
            logging.info("Enable attn fusion ops with normal mode.")
        else:
            logging.info("Enable traditional einsum ops according to your configuration.")

    def forward(
            self,
            x: torch.Tensor,
            x_offsets: torch.Tensor,
            all_timestamps: torch.Tensor,
            attn_mask: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            cache: Optional[List[TransformerCacheState]] = None,
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            return_cache_states: bool = False,
            time_bias: torch.Tensor = torch.tensor([])
    ) -> Tuple[torch.Tensor, List[TransformerCacheState]]:
        """
        前向传播方法, 通过多个STU模块处理输入序列.

        :param x: 输入序列的特征, 形状为(\sum_i N_i, D).
        :param x_offsets: 输入序列的偏移量, 形状为(B + 1).
        :param all_timestamps: 时间戳序列, 形状为(B, 1 + N).
        :param attn_mask: 无效的注意力掩码, 形状为(B, N, N).
        :param past_lengths: 过去序列的长度, 形状为(B,).
        :param num_rerank: 需要推理部分的长度.
        :param cache: 可选参数, 缓存状态列表.
        :param delta_x_offsets: 可选参数, 形状为形状为((B,), (B,))的偏移量.
        :param return_cache_states: 是否返回缓存状态.
        :return: 处理后的输出序列(\sum_i N_i, D), 缓存状态列表.
        """
        cache_states: List[TransformerCacheState] = []
        if self.enable_fusion_ops:
            attn_mask = attn_mask.unsqueeze(1)
            # 训练、评估时mask做过特殊处理，无法直接使用融合算子内置的mask。 repeat至 [bs, _num_heads, n, n]
            attn_mask = attn_mask.repeat(1, self._num_heads, 1, 1)

        for i, layer in enumerate(self._attention_layers):
            with record_function("## hstu or fuxi layer ##"):
                x, cache_states_i, time_bias = layer(
                    x=x,
                    x_offsets=x_offsets,
                    fusion_enabled=self.enable_fusion_ops,
                    jagged_enabled=self.enable_jagged_ops,
                    all_timestamps=all_timestamps,
                    attn_mask=attn_mask,
                    past_lengths=past_lengths,
                    num_rerank=num_rerank,
                    layer_num=i,
                    cache=cache[i] if cache is not None else (
                        torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])),
                    delta_x_offsets=delta_x_offsets,
                    return_cache_states=return_cache_states,
                    time_bias=time_bias
                )
                if return_cache_states:
                    cache_states.append(cache_states_i)

        return x, cache_states
    

@ModelRegistry.register(opt_subs={"RABModule"})
class TransformerEncoderLayer(BaseModel):
    """
    Multi head attention (Transformers) for the behavior sequence.
    """

    """
    基础的 Sequential Transduction Unit, STU 用于处理序列数据.
    """
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        
        # feat_conf
        feat_conf = common_hp.get("feature_conf")
        fuse_ia = feat_conf.get("fuse_ia", False)
        assert fuse_ia, 'fuse_ia must be True when using MultiHeadAttention'

        # model_conf
        model_conf = common_hp.get("model_conf")
        self._embedding_dim: int = model_conf.get("item_embedding_dim", 128)
        self._normalization: str = model_conf.get("normalization", "rel_bias")

        # hp of model_cfg
        sequential_module_config = model_cfg[Const.HP]
        self._num_heads: int = sequential_module_config.get("num_heads", 4)
        self._attention_dim: int = sequential_module_config.get("attention_dim", 32)
        self._attention_dropout: float = sequential_module_config.get('attention_dropout', 0.0)
        self._sublayer_dropout: float = sequential_module_config.get('sublayer_dropout', 0.0)
        self._eps: float = Const.EPS
        self._norm_method: str = sequential_module_config.get('norm_method', 'rms_norm') # 'layer_norm' or 'rms_norm'
        self._norm_position: str = sequential_module_config.get('norm_position', 'pre_norm') # 'pre_norm' or 'post_norm'
        self.ffn_type = sequential_module_config.get('ffn_type', 'glu_ffn') # 'glu_ffn' or 'ffn'
        self.ffn_expand = sequential_module_config.get('ffn_expand', 6)

        self._rel_attn_bias: RABModule = \
            self.init_sub_model("RABModule") if "RABModule" in model_cfg[Const.SUB_MODELS] else None

        # FFN Output
        if self.ffn_type == 'glu_ffn':
            self.feed_forward = GLUFFN(self._embedding_dim, self._embedding_dim, self._embedding_dim, 
                                       ffn_dim_multiplier=self.ffn_expand)
        elif self.ffn_type == 'ffn':
            self.feed_forward = FeedForward(self._embedding_dim, int(self._embedding_dim * self.ffn_expand), 0.0)
        else:
            raise ValueError(f'ffn type {self.ffn_type} is not valid!!!')
        self.dropout_ffn = torch.nn.Dropout(self._sublayer_dropout)

        # Self-Attention
        self.self_attention = ScaledDotProductAttention(
            embedding_dim=self._embedding_dim, 
            num_heads=self._num_heads, 
            attention_dim=self._attention_dim,
            dropout_rate=self._attention_dropout)
        self.dropout_attn = torch.nn.Dropout(self._sublayer_dropout)

        if self._norm_method == 'rms_norm':
            self.attn_layer_norm = RMSNorm_npu(self._embedding_dim, eps=self._eps)
            self.ffn_layer_norm = RMSNorm_npu(self._embedding_dim, eps=self._eps)
        else:
            raise ValueError(f'norm method {self._norm_method} is not valid!!!')
        
        assert self._norm_position in ['pre_norm', 'post_norm'], f'norm_position must be in ["pre_norm", "post_norm"]'

    def forward(
            self,
            x: torch.Tensor,
            x_offsets: torch.Tensor,
            all_timestamps: torch.Tensor,
            attn_mask: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            layer_num: int,
            cache: TransformerCacheState = (torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])),
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            return_cache_states: bool = False,
            time_bias: torch.Tensor = torch.Tensor([])
    ) -> Tuple[torch.Tensor, TransformerCacheState, torch.Tensor]:
        """
        前向传播方法, 处理输入序列并生成输出序列.

        :param hist_x: embeddings of behavior sequence, [B, L, M * D_attr].
        :param cand_x: embeddings of candidates, [B, C, D_i]. 
        :param all_timestamps: 可选参数, 时间戳序列, 形状为(B, N).
        :param attn_mask: [B, C, L], True for valid token, False for invalid token. 
        :param past_lengths: 过去序列的长度, 形状为(B,).
        :param num_rerank: 需要推理部分的长度.
        :param layer_num: 当前层的编号.
        :param cache: 可选参数, 缓存状态, 用于存储中间结果(v, padded_q, padded_k, output).
        :param return_cache_states: 是否返回缓存状态.
        :return: 处理后的输出序列, 形状为(\sum_i N_i, D).
        """
        # n 代表整个需要推理的序列长度

        # logging.info(f'x shape: {x.shape}')

        B = x.shape[0] 
        attn_mask = attn_mask.to(x.device)

        # self-attention
        attn_res = x
        # pre-norm 
        if self._norm_position == 'pre_norm':
            x = self.attn_layer_norm(x) # [B, L, D_in]
        
        if self._normalization == "rel_bias":
            rel_attention_mask = None

            # TODO: get relative attention bias 
            # if all_timestamps is not None and self._rel_attn_bias is not None:
                # # Relative Attention Bias --> attention bias
                # # reshape qk_attn:  batch x num_head x 2n x 2n  --> batch x num_head x n x 2 x n x 2
                # rel_attention_mask, time_bias = self._rel_attn_bias(
                #          all_timestamps, past_lengths, num_rerank,
                #          layer_num, time_bias)
                # # 形如 [bs, _num_heads, (n-1), (n-1)]
                # rel_attention_mask = rel_attention_mask.unsqueeze(1).repeat(1, self._num_heads, 1, self.token_per_item)
                # seq_tokens = n // self.token_per_item - 1
                # rel_attention_mask = rel_attention_mask.view(
                #     B, self._num_heads, seq_tokens, 1, seq_tokens
                # )
                # rel_attention_mask = rel_attention_mask.repeat(1, 1, 1, 1, self.token_per_item)
                # rel_attention_mask = rel_attention_mask.transpose(-1, -2).reshape(B, self._num_heads, n - 1, n - 1)
                # # 形如 [bs, _num_heads, n, n]，补上user
                # rel_attention_mask = torch.nn.functional.pad(rel_attention_mask, (1, 0, 1, 0), 'constant', 0.0) 
            
            # fuse relative attention bias with attention 
            # if HAS_ATTN_FUSION_OPS: # TODO: how to use attn fusion
                # #print("is using attn fusion ops")
                # if num_rerank == 0:
                #     qk_shape = (-1, self._num_heads, self._attention_dim)
                #     v_shape = (-1, self._num_heads, self._linear_dim)
                #     mask = None
                #     mask_type = 0  # 0: tril
                #     layout = "jagged"
                #     seq_offset = x_offsets.tolist()
                #     out_shape = (-1, self._num_heads * self._linear_dim)
                # else:
                #     qk_shape = (B, n, self._num_heads, self._attention_dim)
                #     v_shape = (B, n, self._num_heads, self._linear_dim)
                #     # 推理时mask做过特殊处理，无法直接使用融合算子内置的mask。 repeat至 [bs, _num_heads, n, n]
                #     mask = attn_mask.unsqueeze(1)
                #     mask = mask.repeat(1, self._num_heads, 1, 1)
                #     mask_type = 3  # 3: custom
                #     layout = "normal"
                #     seq_offset = None
                #     out_shape = (B, n, self._num_heads * self._linear_dim)

                # # input:(q, k, v, mask, rab, mask_type, max_seq_len, silu_scale, layout, seq_offset=None)
                # # mask_type: 0 tril, 1 triu, 2 none, 3 custom. layout: "normal" padding, "jagged" non-padding
                
                # attn_output = torch.ops.mxrec.hstu_dense(
                #     q.view(qk_shape), k.view(qk_shape), v.view(v_shape), mask, rel_attention_mask, mask_type,
                #     n, self.qk_attn_denominator_value, layout, seq_offset
                # ).reshape(out_shape)
            # else:
            # logging.info(f'q shape: {q.shape}')
            # logging.info(f'k shape: {k.shape}')
            # logging.info(f'v shape: {v.shape}')
            attn_output, _ = self.self_attention(x, x, x, 
                                                mask=attn_mask, rel_attn_mask=rel_attention_mask) # [B, L, D_in]
            # Add & Norm 
            attn_output = self.dropout_attn(attn_output) + attn_res 
            if self._norm_position == 'post_norm':
                attn_output = self.attn_layer_norm(attn_output)
            # logging.info(f'attn_output shape: {attn_output.shape}')
            
        else:
            raise ValueError("Unknown normalization method %s", self._normalization)

        # ffn 
        ffn_res = attn_output

        ffn_output = attn_output
        # FFN for Q 
        if self._norm_position == 'pre_norm':
            ffn_output = self.ffn_layer_norm(ffn_output)
        ffn_output = self.feed_forward(ffn_output)
        # Add & norm 
        ffn_output = self.dropout_ffn(ffn_output) + ffn_res 
        if self._norm_position == 'post_norm': 
            ffn_output = self.ffn_layer_norm(ffn_output)

        return ffn_output, (x, x, x, ffn_output), time_bias


@ModelRegistry.register()
class CrossAttentionLayer(BaseModel):
    """
    Multi-head target-aware cross attention for the behavior sequence.
    """

    """
    基础的 Sequential Transduction Unit, STU 用于处理序列数据.
    """
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        
        # feat_conf
        feat_conf = common_hp.get("feature_conf")
        fuse_ia = feat_conf.get("fuse_ia", False)
        assert fuse_ia, 'fuse_ia must be True when using MultiCrossAttention'

        # model_conf
        model_conf = common_hp.get("model_conf")
        self._embedding_dim: int = model_conf.get("item_embedding_dim", 128)
        self._normalization: str = model_conf.get("normalization", "rel_bias")

        # hp of model_cfg
        sequential_module_config = model_cfg[Const.HP]
        self._num_heads: int = sequential_module_config.get("num_heads", 4)
        self._attention_dim: int = sequential_module_config.get("attention_dim", 32)
        self._attention_dropout: float = sequential_module_config.get('attention_dropout', 0.0)
        self._sublayer_dropout: float = sequential_module_config.get('sublayer_dropout', 0.0)
        self._eps: float = Const.EPS
        self._norm_method: str = sequential_module_config.get('norm_method', 'rms_norm') # 'layer_norm' or 'rms_norm'
        self._norm_position: str = sequential_module_config.get('norm_position', 'pre_norm') # 'pre_norm' or 'post_norm'
        
        self.num_ffn_layers = sequential_module_config.get('num_ffn_layers', 1) 
        self.ffn_type = sequential_module_config.get('ffn_type', 'glu_ffn') # 'glu_ffn' or 'ffn'
        self.ffn_expand = sequential_module_config.get('ffn_expand', 6)

        self._rel_attn_bias: RABModule = \
            self.init_sub_model("RABModule") if "RABModule" in model_cfg[Const.SUB_MODELS] else None
     
        # FFN
        
        # FFN Output
        if self.ffn_type == 'glu_ffn':
            self.user_ffn_list = nn.ModuleList(
                GLUFFN(self._embedding_dim, self._embedding_dim, self._embedding_dim,
                       ffn_dim_multiplier=self.ffn_expand) for _ in range(self.num_ffn_layers)
            )
            self.hist_ffn_list = nn.ModuleList(
                GLUFFN(self._embedding_dim, self._embedding_dim, self._embedding_dim,
                       ffn_dim_multiplier=self.ffn_expand) for _ in range(self.num_ffn_layers)
            )
            self.cand_ffn_list = nn.ModuleList(
                GLUFFN(self._embedding_dim, self._embedding_dim, self._embedding_dim,
                       ffn_dim_multiplier=self.ffn_expand) for _ in range(self.num_ffn_layers)
            )
        elif self.ffn_type == 'ffn':
            self.user_ffn_list = nn.ModuleList(
                FeedForward(self._embedding_dim, int(self._embedding_dim * self.ffn_expand), 0.0)
                for _ in range(self.num_ffn_layers)
            )
            self.hist_ffn_list = nn.ModuleList(
                FeedForward(self._embedding_dim, int(self._embedding_dim * self.ffn_expand), 0.0)
                for _ in range(self.num_ffn_layers)
            )
            self.cand_ffn_list = nn.ModuleList(
                FeedForward(self._embedding_dim, int(self._embedding_dim * self.ffn_expand), 0.0)
                for _ in range(self.num_ffn_layers)
            )
        else:
            raise ValueError(f'ffn type {self.ffn_type} is not valid!!!')
        self.dropout_ffn = torch.nn.Dropout(self._sublayer_dropout)
        if self._norm_method == 'rms_norm':
            self.user_ffn_ln_list = nn.ModuleList(
                RMSNorm_npu(self._embedding_dim, eps=self._eps)
                for _ in range(self.num_ffn_layers)
            )
            self.hist_ffn_ln_list = nn.ModuleList(
                RMSNorm_npu(self._embedding_dim, eps=self._eps)
                for _ in range(self.num_ffn_layers)
            )
            self.cand_ffn_ln_list = nn.ModuleList(
                RMSNorm_npu(self._embedding_dim, eps=self._eps)
                for _ in range(self.num_ffn_layers)
            )

        # Multi-head Cross Attention
        self.dot_attention = ScaledDotProductAttention(
            embedding_dim=self._embedding_dim, 
            num_heads=self._num_heads, 
            attention_dim=self._attention_dim,
            dropout_rate=self._attention_dropout)
        self.dropout_attn = torch.nn.Dropout(self._sublayer_dropout)

        if self._norm_method == 'rms_norm':
            self.user_attn_layer_norm = RMSNorm_npu(self._embedding_dim, eps=self._eps)
            self.hist_attn_layer_norm = RMSNorm_npu(self._embedding_dim, eps=self._eps)
            self.cand_attn_layer_norm = RMSNorm_npu(self._embedding_dim, eps=self._eps)

    def forward(
            self,
            x: Tuple[torch.Tensor],
            x_offsets: torch.Tensor,
            all_timestamps: torch.Tensor,
            attn_mask: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            layer_num: int,
            cache: TransformerCacheState = (torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])),
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            return_cache_states: bool = False,
            time_bias: torch.Tensor = torch.tensor([])
    ) -> Tuple[torch.Tensor, TransformerCacheState, torch.Tensor]:
        """
        前向传播方法, 处理输入序列并生成输出序列.

        :param x: embeddings of history sequence and candidates, [B, L, D] for history, [B, C, D] for candidates
        :param all_timestamps: 可选参数, 时间戳序列, 形状为(B, N).
        :param attn_mask: [B, C, L], True for valid token, False for invalid token. 
        :param past_lengths: 过去序列的长度, 形状为(B,).
        :param num_rerank: 需要推理部分的长度.
        :param layer_num: 当前层的编号.
        :param cache: 可选参数, 缓存状态, 用于存储中间结果(v, padded_q, padded_k, output).
        :param return_cache_states: 是否返回缓存状态.
        :return: 处理后的输出序列, 形状为(\sum_i N_i, D).
        """
        # n 代表整个需要推理的序列长度
        user_x, hist_x, cand_x = x
        hist_x:torch.Tensor

        B = hist_x.shape[0] 
        attn_mask = attn_mask.to(hist_x.device)
        
        attn_res = cand_x
        # logging.info(f'cand_x shape: {cand_x.shape}')
        # pre-norm 
        if self._norm_position == 'pre_norm':
            user_x = self.user_attn_layer_norm(user_x) # [B, 1, D_i]
            hist_x = self.hist_attn_layer_norm(hist_x) # [B, L, D_i]
            cand_x = self.cand_attn_layer_norm(cand_x) # [B, C, D_i]
        
        if self._normalization == "rel_bias":
            rel_attention_mask = None

            # TODO: get relative attention bias 
            if all_timestamps is not None and self._rel_attn_bias is not None:
                # # Relative Attention Bias --> attention bias
                # # reshape qk_attn:  batch x num_head x 2n x 2n  --> batch x num_head x n x 2 x n x 2
                # rel_attention_mask, time_bias = self._rel_attn_bias(
                #          all_timestamps, past_lengths, num_rerank,
                #          layer_num, time_bias)
                # # 形如 [bs, _num_heads, (n-1), (n-1)]
                # rel_attention_mask = rel_attention_mask.unsqueeze(1).repeat(1, self._num_heads, 1, self.token_per_item)
                # seq_tokens = n // self.token_per_item - 1
                # rel_attention_mask = rel_attention_mask.view(
                #     B, self._num_heads, seq_tokens, 1, seq_tokens
                # )
                # rel_attention_mask = rel_attention_mask.repeat(1, 1, 1, 1, self.token_per_item)
                # rel_attention_mask = rel_attention_mask.transpose(-1, -2).reshape(B, self._num_heads, n - 1, n - 1)
                # # 形如 [bs, _num_heads, n, n]，补上user
                # rel_attention_mask = torch.nn.functional.pad(rel_attention_mask, (1, 0, 1, 0), 'constant', 0.0)
                pass 
            
            # fuse relative attention bias with attention 
            # if HAS_ATTN_FUSION_OPS: # TODO: how to use attn fusion
                # #print("is using attn fusion ops")
                # if num_rerank == 0:
                #     qk_shape = (-1, self._num_heads, self._attention_dim)
                #     v_shape = (-1, self._num_heads, self._linear_dim)
                #     mask = None
                #     mask_type = 0  # 0: tril
                #     layout = "jagged"
                #     seq_offset = x_offsets.tolist()
                #     out_shape = (-1, self._num_heads * self._linear_dim)
                # else:
                #     qk_shape = (B, n, self._num_heads, self._attention_dim)
                #     v_shape = (B, n, self._num_heads, self._linear_dim)
                #     # 推理时mask做过特殊处理，无法直接使用融合算子内置的mask。 repeat至 [bs, _num_heads, n, n]
                #     mask = attn_mask.unsqueeze(1)
                #     mask = mask.repeat(1, self._num_heads, 1, 1)
                #     mask_type = 3  # 3: custom
                #     layout = "normal"
                #     seq_offset = None
                #     out_shape = (B, n, self._num_heads * self._linear_dim)

                # # input:(q, k, v, mask, rab, mask_type, max_seq_len, silu_scale, layout, seq_offset=None)
                # # mask_type: 0 tril, 1 triu, 2 none, 3 custom. layout: "normal" padding, "jagged" non-padding
                
                # attn_output = torch.ops.mxrec.hstu_dense(
                #     q.view(qk_shape), k.view(qk_shape), v.view(v_shape), mask, rel_attention_mask, mask_type,
                #     n, self.qk_attn_denominator_value, layout, seq_offset
                # ).reshape(out_shape)
            # else:
            # logging.info(f'q shape: {q.shape}')
            # logging.info(f'k shape: {k.shape}')
            # logging.info(f'v shape: {v.shape}')
            all_hist_x = torch.concat([user_x, hist_x], dim=1) # [B, L + 1, D_i]
            attn_output, _ = self.dot_attention(
                cand_x, 
                all_hist_x,
                all_hist_x, 
                mask=attn_mask,
                rel_attn_mask=rel_attention_mask) # [B, C, D_i]
            
            # Add & Norm 
            # TODO: whether add residual for user_x and hist_x
            user_x = self.dropout_attn(user_x) + user_x 
            hist_x = self.dropout_attn(hist_x) + hist_x
            attn_output = self.dropout_attn(attn_output) + attn_res 
            if self._norm_position == 'post_norm':
                user_x = self.user_attn_layer_norm(user_x)
                hist_x = self.hist_attn_layer_norm(hist_x)
                attn_output = self.cand_attn_layer_norm(attn_output)
            # logging.info(f'attn_output shape: {attn_output.shape}')
            
        else:
            raise ValueError("Unknown normalization method %s", self._normalization)

        # FFN layers

        user_ffn_output = user_x 
        hist_ffn_output = hist_x
        cand_ffn_output = attn_output
        for i in range(self.num_ffn_layers):
            user_ffn_res = user_ffn_output
            hist_ffn_res = hist_ffn_output
            cand_ffn_res = cand_ffn_output

            # FFN for Q 
            if self._norm_position == 'pre_norm':
                user_ffn_output = self.user_ffn_ln_list[i](user_ffn_output)
                hist_ffn_output = self.hist_ffn_ln_list[i](hist_ffn_output)
                cand_ffn_output = self.cand_ffn_ln_list[i](cand_ffn_output)
            user_ffn_output = self.user_ffn_list[i](user_ffn_output)
            hist_ffn_output = self.hist_ffn_list[i](hist_ffn_output)
            cand_ffn_output = self.cand_ffn_list[i](cand_ffn_output)
            # Add & norm 
            user_ffn_output = self.dropout_ffn(user_ffn_output) + user_ffn_res 
            hist_ffn_output = self.dropout_ffn(hist_ffn_output) + hist_ffn_res 
            cand_ffn_output = self.dropout_ffn(cand_ffn_output) + cand_ffn_res 
            if self._norm_position == 'post_norm': 
                user_ffn_output = self.user_ffn_ln_list[i](user_ffn_output)
                hist_ffn_output = self.hist_ffn_ln_list[i](hist_ffn_output)
                cand_ffn_output = self.cand_ffn_ln_list[i](cand_ffn_output)
            # logging.info(f'ffn_output shape: {ffn_output.shape}')
        
        return (user_ffn_output, hist_ffn_output, cand_ffn_output), \
            (all_hist_x, cand_x, all_hist_x, cand_ffn_output), time_bias


@ModelRegistry.register(req_hp=True, req_subs={"Transformer"})
class SpatialTemporalModule(BaseModel):
    """
    Wide (spatial-temporal):
      - wide_x: F 个独立特征序列 (含主 x)
      - wide_y: 堆叠层数；每层包含: [F 个 temporal(各1层)] -> [1 个 spatial(1层)]
    Deep: 与 WideDeepModule 一致
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        hp = model_cfg[Const.HP]
        model_conf = common_hp["model_conf"]
        self.num_blocks = hp.get("num_blocks", 8)
        self.num_heads = model_cfg[Const.SUB_MODELS]["Transformer"]["hp"].get("num_heads", 4)
        self.enable_fusion_ops = model_cfg[Const.SUB_MODELS]["Transformer"]["hp"].get("enable_fusion_ops", False)
        self.enable_jagged_ops = model_conf.get("enable_jagged_ops", False)


        self.wide_x: int = hp.get("num_wide_x", 3)   # F
        self.wide_y: int = hp.get("num_wide_y", 2)   # N
        self.num_deep_y: int = hp.get("num_deep_y", 1)

        self.fuse_method=model_cfg.get('fuse_subseq', 'add')

        # === 每层都有 F 个 temporal（单层） ===
        self.temporal_layers = torch.nn.ModuleList([
            torch.nn.ModuleList([
                TransformerInner(modules=[self.init_sub_model("Transformer")],
                            num_heads=self.num_heads,
                            enable_fusion_ops=self.enable_fusion_ops,
                            enable_jagged_ops=self.enable_jagged_ops)  # 单层
                for _ in range(self.wide_x)
            ])
            for _ in range(self.wide_y)
        ])

        # === 每层 1 个 spatial（单层） ===
        self.spatial_layers = torch.nn.ModuleList([
            TransformerInner(modules=[self.init_sub_model("Transformer")],
                            num_heads=self.num_heads,
                            enable_fusion_ops=self.enable_fusion_ops,
                            enable_jagged_ops=self.enable_jagged_ops)      # 单层
            for _ in range(self.wide_y)
        ])

        # === F 维融合权重 ===
        self.alpha = torch.nn.Parameter(torch.zeros(self.wide_x))

        # === deep 分支 ===
        self._deep = TransformerInner(
            modules=[self.init_sub_model("Transformer") for _ in range(self.num_deep_y)],
                            num_heads=self.num_heads,
                            enable_fusion_ops=self.enable_fusion_ops,
                            enable_jagged_ops=self.enable_jagged_ops
        )

    def _check_inputs(self, feats: List[torch.Tensor]) -> Tuple[int, int, int]:
        # print(f"HHH:{self.wide_x},{len(feats)}")
        if len(feats) != self.wide_x:
            raise ValueError(f"Expect {self.wide_x} feature sequences, got {len(feats)}.")
        # self.wide_x：4   len(feats)：1
        B, L, D = feats[0].shape
        for i, t in enumerate(feats):
            if t.dim() != 3 or t.shape != (B, L, D):
                raise ValueError(f"Feature[{i}] must be (B,L,D) and match others, got {t.shape}.")
        return B, L, D

    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: torch.Tensor,
        attn_mask: torch.Tensor,
        past_lengths: torch.Tensor,
        num_rerank: int,
        cache: Optional[List[TransformerCacheState]] = None,
        delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
        return_cache_states: bool = False,
        wide_embeddings: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[TransformerCacheState]]:
        # print(f"HHH:{x.shape}")  # ([256, 201, 256]
        feats: List[torch.Tensor] = [x] + ([] if wide_embeddings is None else list(wide_embeddings))
        if self.enable_jagged_ops:
#             B, L, _ = attn_mask.shape
            D = feats[0].shape[1]
        else:
            B, L, D = self._check_inputs(feats)
        F = self.wide_x

        # 为 spatial 步准备一个“无因果”的注意力掩码（跨特征，不需要时间因果）
        spatial_attn_mask = None
        try:
            device = feats[0].device
            spatial_attn_mask = torch.zeros(B * L, F, F, dtype=torch.bool, device=device)
        except Exception:
            spatial_attn_mask = None

        for layer_idx in range(self.wide_y):
            # ---- temporal: 本层的 F 个单层 Transformer，分别处理各自特征 ----
            tmp: List[torch.Tensor] = []
            for f in range(F):
                y, _ = self.temporal_layers[layer_idx][f](
                    x=feats[f],
                    x_offsets=x_offsets,
                    all_timestamps=all_timestamps,
                    attn_mask=attn_mask,                 # 时间掩码/因果
                    past_lengths=past_lengths,
                    num_rerank=num_rerank,
                    cache=cache,
                    delta_x_offsets=delta_x_offsets,
                    return_cache_states=False,
                )
                tmp.append(y)
            feats = tmp  # F * (B,L,D)

            # ---- spatial: 使用rankmixer----
            H = F
            assert D % H == 0, f"D ({D}) 必须能被 H=F ({H}) 整除"
            d_head = D // H
            if self.enable_jagged_ops:
                # stacked = torch.stack(feats, dim=2)           # (B, L, F, D)
                stacked = torch.stack(feats, dim=1) 

                # (B, L, F, H, D//H)
                stacked = stacked.view(-1, F, H, d_head)

                # -> (B, L, H, F, D//H)
                stacked = stacked.permute(0, 2, 1, 3).contiguous()

                # -> (B, L, H, F*D//H), 即 （B, L, F, D）
                stacked = stacked.view(-1, H, F * d_head)

                feats = [stacked[:, f, :] for f in range(F)]
            else:
                stacked = torch.stack(feats, dim=2)           # (B, L, F, D)

                # (B, L, F, H, D//H)
                stacked = stacked.view(B, L, F, H, d_head)

                # -> (B, L, H, F, D//H)
                stacked = stacked.permute(0, 1, 3, 2, 4).contiguous()

                # -> (B, L, H, F*D//H), 即 （B, L, F, D）
                stacked = stacked.view(B, L, H, F * d_head)

                feats = [stacked[:, :, f, :] for f in range(F)]

        # ---- 融合多个序列：softmax(alpha) -> (B,L,D) ----
        if self.fuse_method=='weight':
            w = torch.softmax(self.alpha, dim=0)               # (F,)
            fused = sum(feats[f] * w[f] for f in range(F))     # (B, L, D)
        else:
            fused= sum(feats[f] for f in range(F))

        # ---- deep ----
        out_x, cache_states = self._deep(
            x=fused,
            x_offsets=x_offsets,
            all_timestamps=all_timestamps,
            attn_mask=attn_mask,
            past_lengths=past_lengths,
            num_rerank=num_rerank,
            cache=cache,
            delta_x_offsets=delta_x_offsets,
            return_cache_states=return_cache_states,
        )

        return out_x, cache_states

    def debug_str(self) -> str:
        return f"SpatialTemporalModule(wide_x={self.wide_x}, wide_y={self.wide_y}, deep_y={self.num_deep_y})"
