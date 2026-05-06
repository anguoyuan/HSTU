from typing import Dict
from datetime import datetime
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.generic.sequential.base_model import BaseModel
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const, FeatConst


@ModelRegistry.register(multi_sel_multi_subs=[{"CausalAttentionMask", "TimeAttentionMask", "MTAttentionMask"}])
class AttentionMaskModule(BaseModel):
    """
    "AttentionMaskModule": {
        "type": ["CausalAttentionMask", "TimeAttentionMask"],
        "cfg": {}
    }

    :param model_conf:
    :param model_factory:
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:

        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self.attention_mask_modules = nn.ModuleList([
            self.init_sub_model(sub_key)
            for sub_key in model_cfg[Const.SUB_MODELS].keys()
        ])

    def forward(self, model_inputs: Dict[str, torch.Tensor], max_seq_len, num_rerank, jagged_enable) -> torch.Tensor:
        init_mask = None
        for attn_module in self.attention_mask_modules:
            mask = attn_module(model_inputs=model_inputs, max_seq_len=max_seq_len, num_rerank=num_rerank, jagged_enable=jagged_enable)
            if init_mask is None:
                init_mask = mask
            else:
                init_mask = init_mask * mask
        attn_mask = init_mask.detach().clone()

        return attn_mask


@ModelRegistry.register()
class CausalAttentionMask(BaseModel):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        """
        因果注意力掩码生成
        训练时：生成形状为(bs, 2 * max_seq_len + 2, 2 * max_seq_len + 2)的下三角矩阵
        推理时：生成形状为(bs, 2 * max_seq_len + 2 + num_rerank, 2 * max_seq_len + 2 + num_rerank)的掩码矩阵，
        其中前2 * max_seq_len + 2行/列与训练时的掩码矩阵相同，但候选集部分token互相不可见
        """
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        feat_conf = common_hp.get("feature_conf")
        self.mask_candidates = model_cfg[Const.HP].get("mask_candidates", True)

        fuse_ia = feat_conf.get("fuse_ia", False)
        self.token_per_item = 2 if not fuse_ia else 1

    def init_mask_for_export(self, seq_len, num_rerank, device):
        max_len = seq_len * self.token_per_item + 1 + num_rerank
        _pos_indices = torch.arange(max_len).repeat(max_len).view(max_len, max_len).to(device)
        _base_mask = (_pos_indices.t() > _pos_indices).float()
        _identity = (_pos_indices.t() == _pos_indices).float()
        return _pos_indices, _identity, _base_mask

    def forward(self, model_inputs: Dict[str, torch.Tensor], max_seq_len, num_rerank, jagged_enable):
        device = model_inputs[self.candidate_timestamps_key].device
        max_len = max_seq_len * self.token_per_item + num_rerank + 1
        if not self.mask_candidates:
            indices = torch.arange(max_len).to(device)
            t = indices.expand(max_len, max_len)
            attn_mask = (t.t() >= indices).unsqueeze(0)

        else:
            if num_rerank == 0:
                max_len = max_seq_len * self.token_per_item + 1
                indices = torch.arange(max_len).to(device)
                t = indices.expand(max_len, max_len)
                attn_mask = (t.t() >= indices).unsqueeze(0)
            else:
                past_lengths = model_inputs['past_lengths']
                past_lengths = 1 + past_lengths * self.token_per_item
                _past_lengths = past_lengths.unsqueeze(-1).unsqueeze(-1)
                _pos_indices, _identity, _base_mask = self.init_mask_for_export(
                    seq_len=max_seq_len, num_rerank=num_rerank, device=device
                )
                seq_mask = (_pos_indices < _past_lengths).int()
                attn_mask = (seq_mask * _base_mask + _identity)

        return attn_mask


@ModelRegistry.register()
class PastCausalAttentionMask(CausalAttentionMask):

    def forward(self, model_inputs, max_seq_len, num_rerank, jagged_enable):
        """
        Returns a bs x (2 * max_seq_len + 2) x (2 * max_seq_len + 2) attn mask
        - [1:idx+1, 1:idx+1] 全可见
        - [idx+1:, idx+1:] 下三角（含对角线）
        where idx = non_zero_index per sample.
        """

        device = model_inputs['past_end_index'].device
        max_len = max_seq_len * self.token_per_item + 2 + num_rerank

        # 每行第一个 1 的位置，即1y-30d这一段；全 0 时应设为 max_len-1 或提前处理为 max_len-1
        past_end_index = model_inputs['past_end_index'] * self.token_per_item

        bs = past_end_index.shape[0]

        row_idx = torch.arange(max_len, device=device).view(1, max_len, 1).expand(bs, -1, -1)
        col_idx = torch.arange(max_len, device=device).view(1, 1, max_len).expand(bs, -1, -1)

        nz = past_end_index.view(bs, 1, 1)

        # 前半部分：row, col 都 ≤ nz，全可见
        pre_mask = (row_idx <= nz) & (col_idx <= nz)

        # 后半部分：row, col 都 ≥ nz+1 且 row ≥ col （下三角），除了最后num_rerank个
        r_start = max_len - num_rerank
        post_mask = (row_idx > nz) & (row_idx < r_start) & (row_idx >= col_idx)

        # 最后num_rerank：能看见自己以及之前num_rerank之前所有的
        rerank_mask = (row_idx >= r_start) & (
                (col_idx < r_start) |
                (row_idx == col_idx)
        )

        # 合并并返回 float mask
        attn_mask = (pre_mask | post_mask | rerank_mask).float()

        attn_mask[:, 0, :] = 0.
        attn_mask[:, 0, 0] = 1.

        return attn_mask


@ModelRegistry.register()
class TimeAttentionMask(BaseModel):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        """
        时间注意力掩码生成
        生成形状为(bs, 2 * max_seq_len + 2 + num_rerank, 2 * max_seq_len + 2 + num_rerank)的掩码矩阵，
        其中每个token的可见范围由timestamp_mask_threshold确定。
        """
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        feat_conf = common_hp.get("feature_conf")
        self._time_threshold = model_cfg[Const.HP].get('timestamp_mask_threshold', 86400)
        self._infer_items_key = feat_conf.get('infer_items_key', 'item_id')
        self._infer_timestamps_key = feat_conf.get("infer_timestamps_key", "timestamps")

        fuse_ia = feat_conf.get("fuse_ia", False)
        self.token_per_item = 2 if not fuse_ia else 1

    def forward(self, model_inputs: Dict[str, torch.Tensor], max_seq_len, num_rerank, jagged_enable):
        device = model_inputs[self.candidate_timestamps_key].device
        bs = model_inputs.get(self.candidate_items_key).shape[0]
        all_timestamps = model_inputs[self._infer_timestamps_key].detach()

        time_threshold_mask = (
                (all_timestamps.unsqueeze(2) - all_timestamps.unsqueeze(1)) <= self._time_threshold
        ).to(torch.float32)

        # 对角线置0
        _pos_indices = torch.arange(max_seq_len).repeat(max_seq_len).view(max_seq_len, max_seq_len).to(device)
        _identity = (_pos_indices.t() == _pos_indices).float()
        time_threshold_mask = time_threshold_mask - _identity

        attn_mask = (
                1.0
                - time_threshold_mask
                .unsqueeze(1).unsqueeze(-1)
                .repeat(1, 1, 1, self.token_per_item, self.token_per_item)
                .reshape(bs, max_seq_len * self.token_per_item, max_seq_len * self.token_per_item)
        )

        attn_mask = F.pad(attn_mask, (1, 1 + num_rerank, 1, 1 + num_rerank), 'constant', 1.0)

        return attn_mask


@ModelRegistry.register()
class MTAttentionMask(BaseModel):
    """
    Meituan方案mask，history部分causal，candidate互相不可见，仅可看见history时间位于其前面的item
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        feat_conf = common_hp.get("feature_conf")
        self.hist_dates_key = feat_conf.get("history_date_column", FeatConst.DFLT_HIST_DATE_KEY)
        self.cand_dates_key = feat_conf.get("candidate_date_column", FeatConst.DFLT_CAND_DATE_KEY)

    def forward(self, model_inputs: Dict[str, torch.Tensor], max_seq_len: int, num_rerank: int, jagged_enable) -> torch.Tensor:
        """
            Returns a bs x (h+p+1) x (h+p+1)  attn mask
        """
        hist_ts = model_inputs[self.hist_dates_key]
        hist_ts = hist_ts[:, -max_seq_len : ]
        cand_ts = model_inputs[self.cand_dates_key]
        cand_ts = cand_ts[:, -num_rerank : ]

        history_lengths = model_inputs.get("history_lengths")
        candidate_lengths = model_inputs.get("candidate_lengths")
        bs, hist_len = hist_ts.shape
        _, cand_len = cand_ts.shape
        max_len = hist_len + cand_len + 1
        device = hist_ts.device
        lengthsum = history_lengths + candidate_lengths
        history_lengths = history_lengths.view(bs, 1, 1)
        lengthsum = lengthsum.view(bs, 1, 1)

        # final_mask 示意图 (1=True, 0=False), h4 和 p3为 padding
        #
        #       |  u | h0 | h1 | h2 | h3 | h4 | p0 | p1 | p2 | p3 |
        #       --------------------------------------------------
        #    u  |  1 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
        #   h0  |  1 |  1 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
        #   h1  |  1 |  1 |  1 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
        #   h2  |  1 |  1 |  1 |  1 |  0 |  0 |  0 |  0 |  0 |  0 |
        #   h3  |  1 |  1 |  1 |  1 |  1 |  0 |  0 |  0 |  0 |  0 |
        #   h4  |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
        #   p0  |  1 |  1 |  1 |  1 |  0 |  0 |  1 |  0 |  0 |  0 |
        #   p1  |  1 |  1 |  1 |  1 |  1 |  0 |  0 |  1 |  0 |  0 |
        #   p2  |  1 |  1 |  1 |  1 |  1 |  0 |  0 |  0 |  1 |  0 |
        #   p3  |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |

        user_col = torch.zeros(bs, 1, device=device, dtype=hist_ts.dtype)
        seq_ts = torch.cat([user_col, hist_ts, cand_ts], dim=1)
        pad_len = max_len - seq_ts.shape[1]
        ts_pad = F.pad(seq_ts, (0, pad_len), mode='constant', value=0)

        idx = torch.arange(max_len, device=device)
        indices = idx.unsqueeze(0) #1*551
        valid_mask = (seq_ts != 0) | (indices == 0)
        past_sum = torch.sum(valid_mask, dim=1).unsqueeze(-1)#B*1
        sort_mask = (indices < past_sum)
        row_idx = idx.view(1, max_len, 1)
        col_idx = idx.view(1, 1, max_len)
        if jagged_enable:
            ts_pad[sort_mask] = ts_pad[valid_mask]#重排 #128*551
            ts_i = ts_pad.unsqueeze(2)
            ts_j = ts_pad.unsqueeze(1)

            # 四个区域
            jagged_region = (row_idx <= lengthsum)
            user_mask = jagged_region & (col_idx == 0)
            hist_region = (col_idx >= 1) & (col_idx < 1 + history_lengths) & (row_idx < history_lengths + 1)
            hist_mask = hist_region & (row_idx >= col_idx)
            cand_region = jagged_region & (row_idx >= 1 + history_lengths) & (col_idx >= 1) & (col_idx < 1 + history_lengths)
            cand_hist_mask = cand_region & (ts_i > ts_j)#B*551*511
            pred_diag_mask = jagged_region & (row_idx == col_idx)

            # 合并 已验证和reorganized版本mask一致
            mask = (user_mask | hist_mask | cand_hist_mask | pred_diag_mask).float()

        else:
            ts_i = ts_pad.unsqueeze(2)
            ts_j = ts_pad.unsqueeze(1)
            valid_row = (ts_i > 0) | (row_idx == 0)
            valid_col = (ts_j > 0) | (col_idx == 0)
            # 四个区域
            user_mask = (col_idx == 0)
            hist_region = (col_idx >= 1) & (col_idx < 1 + hist_len) & (row_idx < hist_len + 1) & (row_idx >= 1)
            hist_mask = hist_region & (row_idx >= col_idx)
            cand_region = (row_idx >= 1 + hist_len) & (col_idx >= 1) & (col_idx < 1 + hist_len)
            cand_hist_mask = cand_region & (ts_i > ts_j)
            pred_diag_mask = (row_idx == col_idx) & (row_idx >= 1 + hist_len)

            # 合并
            mask = ((user_mask | hist_mask | cand_hist_mask | pred_diag_mask) & valid_row & valid_col).float()

        # mask_np = mask.detach().cpu().numpy()
        # B, N, _ = mask_np.shape

        # import matplotlib.pyplot as plt
        # import os
        # for i in range(B):
        #     plt.figure(figsize=(4, 4))
        #     plt.imshow(mask_np[i], cmap='plasma', interpolation='nearest')
        #     # 保存文件
        #     filename = os.path.join("/opt/huawei/dataset/data_dir/20250610/model", f"mask_{i}.png")
        #     plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        #     plt.close()
        #     logging.info("Saved: %s", filename)
        # raise ValueError

        return mask
