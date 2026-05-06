from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Optional, Union

import torch
import torchrec
from torchrec import JaggedTensor, KeyedJaggedTensor
import functools
import torch.nn.functional as F
from collections import ChainMap
from modeling.generic.sequential.attn_mask_modules import AttentionMaskModule
from modeling.generic.sequential.base_model import BaseModel
from modeling.generic.sequential.embedding_modules import EmbeddingModule, EmbeddingType, InitEmbeddingConfig
from modeling.generic.sequential.input_features_preprocessors import InputFeaturesPreprocessorModule
from modeling.generic.sequential.loss_modules import LossModule
from modeling.generic.sequential.negative_sampler import NegativesSampler
from modeling.generic.sequential.output_postprocessors import OutputPostprocessorModule
from modeling.generic.sequential.prediction_modules import FeedForwardModule
from modeling.generic.sequential.transformers import SequentialModule, TransformerCacheState
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const, FeatConst
from modeling.generic.utils.reshape import reorganize_tensors, extract_ranked_embeddings
from modeling.generic.sequential.dlrm_modules import CrossNetwork, PPNetLayer, MLPLayer, MaskedBatchNorm1d
from modeling.generic.initialization import truncated_normal
from modeling.generic.utils.jagged_utils import jagged_to_padded_dense, dense_to_jagged
from torch.autograd.profiler import record_function


class initializing:
    def __init__(self, enter_fn: Callable[[], None] = None, exit_fn: Callable[[], None] = None):
        self.enter_fn = enter_fn
        self.exit_fn = exit_fn

    def __enter__(self):
        if callable(self.enter_fn):
            self.enter_fn()

    def __exit__(self, *exc_info):
        if callable(self.exit_fn):
            self.exit_fn()


@ModelRegistry.register()
class LinearModuleForRerankScore(BaseModel):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self.name = "rerank_score"
        self.input_dim = model_cfg['hp']['input_dim']
        self.pred_linear = torch.nn.Linear(in_features=self.input_dim, out_features=1)

    def forward(self, x: torch.Tensor):
        x = self.pred_linear(x)
        x = torch.sigmoid(x)
        return self.name, x


@ModelRegistry.register()
class DLRM_model_v2(BaseModel):

    def __init__(self, model_cfg, common_hp, model_cls_dict):
        super().__init__(model_cfg, common_hp, model_cls_dict)
        model_hp = model_cfg.get('hp')
        self.feature_conf = common_hp.get("feature_conf")
        self.feature_groups = self.feature_conf.get("feature_groups")
        self.hist_items_key = self.feature_conf.get("history_items_key", FeatConst.DFLT_HIST_ITEM_KEY)
        self.cand_items_key = self.feature_conf.get("candidate_items_key", FeatConst.DFLT_CAND_ITEM_KEY)
        self.enable_jagged_ops = common_hp["model_conf"].get("enable_jagged_ops", False) and self.training

        # use feature embedding from sequential model
        self.embedding_module: EmbeddingModule = model_cfg['embedding_module']
        self.seq_input_dim = model_cfg['item_embedding_dim']
        logging.info(f'seq_input_dim in DLRM : {self.seq_input_dim}')

        self.use_dlrm_can = model_hp.get('use_dlrm_can', True)
        self.can_as_dcn_input = model_hp.get('can_as_dcn_input', True)
        self.can_as_dnn_input = model_hp.get('can_as_dnn_input', True)
        self.can_as_final_input = model_hp.get('can_as_final_input', True)
        self.use_dlrm_linear = model_hp.get('use_dlrm_linear', True)
        self.use_dlrm_cross = model_hp.get('use_dlrm_cross', True)
        self.use_dlrm_dnn = model_hp.get('use_dlrm_dnn', True)
        self.use_dlrm_ppnet = model_hp.get('use_dlrm_ppnet', True)

        self.seq_as_cross_input = model_hp.get('seq_as_cross_input', True)
        self.seq_as_dnn_input = model_hp.get('seq_as_dnn_input', True)
        self.seq_as_ppnet_input = model_hp.get('seq_as_ppnet_input', True)
        self.seq_as_final_input = model_hp.get('seq_as_final_input', True)
        # self.con_feature_tables = torch.nn.ModuleDict()

        # get input dim
        self.cand_feature_dim = self._calculate_input_dim(self.feature_groups['candidate_dlrm']['features'])
        self._initialize_con()
        # self.masked_bn = MaskedBatchNorm1d()

        # init feature interaction modules
        ## CAN
        if self.use_dlrm_can:
            self.can_input_dim = model_hp.get('can_input_dim', 16)
            self.can_hidden_dims = model_hp.get('can_hidden_dims', [8, 4])
            self.can_selected_features = model_hp.get('can_selected_features',
                                                      self.feature_groups['candidate_dlrm']['features'])
            self.cand_can_dim = self._calculate_can_input_dim(self.can_selected_features)
            self.emb_can_table_single: Dict[str, torchrec.EmbeddingConfig] = {}
            self.emb_can_table_multi: Dict[str, torchrec.EmbeddingConfig] = {}
            self._initialize_features_can_embeddings(self.can_selected_features)
            self._create_features_can_ecs()
            logging.info(f'can_input_dim: {self.cand_can_dim}')
            logging.info(f'can_selected_features: {self.can_selected_features}')

        ## DCN
        if self.use_dlrm_cross:
            self.num_cross_layers = model_hp.get('num_cross_layers', 3)
            # input_dim_cross = 32088 + 2560
            input_dim_cross = self.cand_feature_dim
            if self.use_dlrm_can and self.can_as_dcn_input:
                input_dim_cross += self.cand_can_dim
            if self.seq_as_cross_input:
                input_dim_cross += self.seq_input_dim
            self.cross = CrossNetwork(input_dim_cross, self.num_cross_layers)

        ## DNN
        if self.use_dlrm_dnn:
            self.dnn_hidden_layers = model_hp.get('dnn_hidden_layers', [1024, 512, 256])
            input_dim_dnn = self.cand_feature_dim
            if self.use_dlrm_can and self.can_as_dnn_input:
                input_dim_dnn += self.cand_can_dim
            if self.seq_as_dnn_input:
                input_dim_dnn += self.seq_input_dim
            self.dnn = MLPLayer(
                hidden_layers=[input_dim_dnn] + self.dnn_hidden_layers,
                act_func=model_hp.get('dnn_act_func', 'relu'),
                dropout_rate=model_hp.get('dnn_dropout_rate', 0.1),
                use_bn=model_hp.get('dnn_use_bn', False)
            )

        ## linear
        if self.use_dlrm_linear:
            self.dlrm_linear = torch.nn.Linear(self.cand_feature_dim, 1, bias=False)

        ## PPNet
        if self.use_dlrm_ppnet:
            self.ppnet_gate_features = model_hp.get('ppnet_gate_features',
                                                    ["candidate_city",
                                                     "candidate_udid_appid_down_list_23d",
                                                     "candidate_uid_appid_down_list_30d",
                                                     "candidate_udid_appid_clicknotdownload_list_14d",
                                                     "candidate_udid_hiad_appid_browse_list_14d",
                                                     "candidate_udid_bymaxcnt_appid_used_list_30d",
                                                     "candidate_udid_appid_uninstall_list_14d",
                                                     "candidate_udid_search_appid_down_list_30d",
                                                     "candidate_udid_appid_used_list_7d",
                                                     "candidate_ubr_down_app_list", "candidate_ubr_used_app_list",
                                                     "candidate_ubr_uninstall_app_list",
                                                     "candidate_udid_extend_appc3_appid_down_list_1y",
                                                     "candidate_uid_appc2_bymaxcnt_appid_used_list_1y",
                                                     "candidate_udid_hourbucket_appid_click_list_1y",
                                                     "candidate_udid_hourbucket_appid_down_list_1y"]
                                                    )
            self.input_ppnet_gate_dim = self._calculate_ppnet_gate_input_dim()
            #             print('self.input_ppnet_gate_dim', self.input_ppnet_gate_dim)  # 7296
            self.ppnet_mlp_layer_dims = model_hp.get('ppnet_mlp_layer_dims', [256, 128])
            self.ppnet_gate_hidden_dims = model_hp.get('ppnet_gate_hidden_dims',
                                                       [self.input_ppnet_gate_dim // 2, self.input_ppnet_gate_dim // 2])
            self.ppnet_dropout_rate = model_hp.get('ppnet_dropout_rate', 0.1)
            self.ppnet_use_bn = model_hp.get('ppnet_use_bn', True)

            input_dim_ppnet = self.cand_feature_dim
            if self.seq_as_ppnet_input:
                input_dim_ppnet += self.seq_input_dim
            self.ppnet = PPNetLayer(
                feature_emb_dim=input_dim_ppnet,  # 特征嵌入维度
                gate_emb_dim=self.input_ppnet_gate_dim,  # 门控嵌入维度
                mlp_layer_dims=self.ppnet_mlp_layer_dims,  # MLP 层维度
                gate_layer_dims=self.ppnet_gate_hidden_dims,  # 门控层维度
                batch_norm=self.ppnet_use_bn,
                dropout_rate=self.ppnet_dropout_rate
            )

        ## final linear
        self.final_input_dim = 0
        if self.use_dlrm_linear:
            self.final_input_dim += 1
        #             print(f'linear dim: 1')
        if self.use_dlrm_can and self.can_as_final_input:
            self.final_input_dim += self.cand_can_dim
        #             print(f'cand_can_dim: {self.cand_can_dim}')
        if self.use_dlrm_cross:
            self.final_input_dim += input_dim_cross
        #             print(f'input_dim_cross: {input_dim_cross}')
        if self.use_dlrm_dnn:
            self.final_input_dim += self.dnn_hidden_layers[-1]
        #             print(f'final dnn dim: {self.dnn_hidden_layers[-1]}')
        if self.use_dlrm_ppnet:
            self.final_input_dim += self.ppnet_mlp_layer_dims[-1]
        #             print(f'final ppnet dim: {self.ppnet_mlp_layer_dims[-1]}')
        if self.seq_as_final_input:
            self.final_input_dim += self.seq_input_dim
        #             print(f'final seq input dim: {self.seq_input_dim}')

        self.reset_params()

    def _calculate_input_dim(self, feature_names: List) -> int:
        input_dim = 0
        for feat_name in feature_names:
            if self.embedding_module._feature_dtypes.get(feat_name) == "con":
                input_dim += 1
            else:
                input_dim += self.embedding_module._feature_dims.get(feat_name, 0)

        return input_dim

    def _calculate_can_input_dim(self, feature_names: List) -> int:
        input_dim = 0
        num_one_hot = 0
        num_multi_hot = 0
        for feat_name in feature_names:
            feat_dtype = self.embedding_module._feature_dtypes.get(feat_name)
            if feat_dtype == "con":  # 这个没处理
                input_dim += 1
            elif feat_dtype == "int" or feat_dtype == "context":
                num_one_hot += 1
            elif feat_dtype == "multi":
                num_multi_hot += self.embedding_module._feats_max_len[feat_name]
        print('CAN features: num_one_hot, num_multi_hot', num_one_hot, num_multi_hot)
        input_dim = num_one_hot * num_multi_hot * sum(self.can_hidden_dims)
        return input_dim

    def _initialize_con(self):

        for feature_name, feature_info in self.embedding_module.all_feature_columns.items():
            feature_enabled = feature_info.get("enabled", True)

            if not feature_enabled:
                continue

            feature_dtype = feature_info.get("dtype", FeatConst.DFLT_DTYPE)
            feature_max_len = feature_info.get("max_len", 1)

            base_feature_name = self.embedding_module._get_base_feature_name(feature_name)

            if feature_dtype == "con":
                setattr(self, f"con_table_{base_feature_name}", MaskedBatchNorm1d())
                # self.con_feature_tables[base_feature_name] = MaskedBatchNorm1d()
            else:
                continue

    def _initialize_features_can_embeddings(self, feature_names: list[str]) -> None:
        # all feature_columns
        all_feature_columns = \
            ["candidate_item_feature_columns", "history_item_feature_columns", "user_feature_columns"]

        # single feature dim
        single_emb_dim = 0
        for in_dim, out_dim in zip(([self.can_input_dim] + self.can_hidden_dims)[:-1], \
                                   ([self.can_input_dim] + self.can_hidden_dims)[1:]):
            single_emb_dim += in_dim * out_dim

        for feature_name in feature_names:
            for feature_column in all_feature_columns:
                group_feature_conf: dict = self.feature_conf.get(feature_column, None)
                if group_feature_conf is None:
                    return
                if feature_name in group_feature_conf:
                    feature_info = group_feature_conf[feature_name]
                    base_feature_name = self.embedding_module._get_base_feature_name(feature_name)

                    feature_count = feature_info.get('feature_count', FeatConst.FEAT_CNT)
                    feature_enabled = feature_info.get("enabled", True)
                    feature_dtype = feature_info.get("dtype", FeatConst.DFLT_DTYPE)

                    if feature_enabled:
                        if feature_dtype == "con":  # TODO
                            pass
                        elif feature_dtype == "multi":
                            #                             print(f'multi: {base_feature_name}')
                            # use shared feature name
                            shared_feat_name = feature_info.get("shared_feat_name", "")
                            if shared_feat_name not in self.embedding_module.all_feature_columns:
                                raise ValueError("multifeat %s, shared_feat_name \"%s\" not found in feature columns",
                                                 feature_name, shared_feat_name)
                            base_shared_feature_name = self.embedding_module._get_base_feature_name(shared_feat_name)
                            if base_shared_feature_name in self.emb_can_table_multi:
                                self.emb_can_table_multi[base_shared_feature_name].feature_names.append(feature_name)
                            else:
                                feature_count = group_feature_conf.get(shared_feat_name) \
                                    .get('feature_count', FeatConst.FEAT_CNT)
                                table = InitEmbeddingConfig(
                                    name=base_shared_feature_name,
                                    embedding_dim=self.can_input_dim,
                                    num_embeddings=feature_count + 1,
                                    feature_names=[feature_name]
                                )
                                self.emb_can_table_multi[base_shared_feature_name] = table
                        elif feature_dtype == "int" or feature_dtype == "context":
                            #                             print(f'single: {base_feature_name}')
                            if base_feature_name in self.emb_can_table_single:
                                self.emb_can_table_single[base_feature_name].feature_names.append(feature_name)
                                continue
                            table = InitEmbeddingConfig(
                                name=base_feature_name,
                                embedding_dim=single_emb_dim,
                                num_embeddings=feature_count + 1,
                                feature_names=[feature_name]
                            )
                            self.emb_can_table_single[base_feature_name] = table
                        else:
                            logging.error("feature_dtype %s is undefined for %s.", feature_dtype, feature_name)
                            pass
                    break

    def _create_features_can_ecs(self):

        # multi
        dim2tables_can_multi: Dict[str, List[torchrec.EmbeddingConfig]] = {}
        self.dim2feats_can_multi: Dict[str, List[str]] = {}
        for table in self.emb_can_table_multi.values():
            # table按dim分类：每个ec中的table.embedding_dim必须相同
            ec_attr = f"ec_can_multi_dim{table.embedding_dim}"
            dim2tables_can_multi.setdefault(ec_attr, []).append(table)
            self.dim2feats_can_multi.setdefault(ec_attr, []).extend(table.feature_names)
        self.feat2dim_can_multi: Dict[str, str] = {}
        for attr, feature_names in self.dim2feats_can_multi.items():
            for name in feature_names:
                self.feat2dim_can_multi[name] = attr
        # 创建EmbeddingCollections
        for attr, tables in dim2tables_can_multi.items():
            setattr(self, attr, torchrec.EmbeddingCollection(device=torch.device("meta"), tables=tables))

        # single
        dim2tables_can_single: Dict[str, List[torchrec.EmbeddingConfig]] = {}
        self.dim2feats_can_single: Dict[str, List[str]] = {}
        for table in self.emb_can_table_single.values():
            # table按dim分类：每个ec中的table.embedding_dim必须相同
            ec_attr = f"ec_can_single_dim{table.embedding_dim}"
            dim2tables_can_single.setdefault(ec_attr, []).append(table)
            self.dim2feats_can_single.setdefault(ec_attr, []).extend(table.feature_names)
        self.feat2dim_can_single: Dict[str, str] = {}
        for attr, feature_names in self.feat2dim_can_single.items():
            for name in feature_names:
                self.feat2dim_can_single[name] = attr
        # 创建EmbeddingCollections
        for attr, tables in dim2tables_can_single.items():
            setattr(self, attr, torchrec.EmbeddingCollection(device=torch.device("meta"), tables=tables))

    def reset_params(self):
        for name, module in self.named_modules():
            if ('embedding_module' in name) or ('ec_can_multi' in name) or ('ec_can_single' in name):
                continue
            if isinstance(module, torch.nn.Embedding):
                truncated_normal(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    torch.nn.init.constant_(module.weight.data[module.padding_idx], 0.)
                logging.info(f"Initialize module {name} as truncated normal: {module.weight.data.size()} params")

            elif isinstance(module, torch.nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()
                logging.info(f"Initialize module {name} with normal(0, 0.01) for weight and zero for bias")

            elif isinstance(module, torch.nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
                logging.info(f"Initialize torch.nn.LayerNorm {name}")
            else:
                logging.info(f"Skipping initializing module {name} - not configured")

        # for nn.Parameter
        for name, param in self.named_parameters():
            # TODO: when the name of nn.Parameter is not 'weight' either 'bias', it won't be skipped.
            all_module_names = [module_name for module_name, _ in self.named_modules() if module_name]
            if not (name.removesuffix('.weight') in all_module_names or \
                    name.removesuffix('.bias') in all_module_names):  # for DCN
                param.data.normal_(mean=0.0, std=0.01)
                logging.info(f"Initialize param {name} with normal(0, 0.01)")

    def _calculate_ppnet_gate_input_dim(self) -> int:
        input_dim = 0

        for feat_name in self.feature_groups['candidate_dlrm']['features']:
            #             print('ppnet dim feat_name', feat_name)
            if feat_name in self.ppnet_gate_features:
                feat_dtype = self.embedding_module._feature_dtypes.get(feat_name)
                if feat_dtype == "con":  # TODO: continous features in PPNet
                    input_dim += 1
                elif feat_dtype == "int" or feat_dtype == 'context':
                    input_dim += self.embedding_module._feature_dims.get(feat_name, 0)
                elif feat_dtype == "multi":  # TODO: pooling sequence features in PPNet
                    input_dim += self.embedding_module._feature_dims.get(feat_name, 0)
        return input_dim

    def _get_feature_embeddings(
            self,
            model_input: Dict[str, torch.Tensor],
            feature_names: List[str],
    ) -> torch.Tensor:
        '''
        model_input: Dict of [B, L] or [B, L, M] or [B, C] or [B, C, M] or [B]
        '''
        feature_emb_list = []
        for feature_name in feature_names:
            feature_id = model_input[feature_name]
            feature_dtype = self.embedding_module._feature_dtypes.get(feature_name)
            base_feature_name = self.embedding_module._get_base_feature_name(feature_name)

            if feature_dtype == "con":
                # feature_value = self.con_feature_tables[base_feature_name](feature_id)
                # module = self.embedding_module.con_feature_tables[base_feature_name]
                module = getattr(self, f"con_table_{base_feature_name}")
                feature_value = module(feature_id)
            else:
                feature_value = self.embedding_module.feature_values_cache[feature_name].values()
                if feature_dtype == "multi":
                    multi = feature_id.size(-1)
                    D = feature_value.size(-1)
                    feature_value = feature_value.reshape(-1, multi, D)  # (N, M, D)
                    feature_id_jagged = \
                        model_input[self.embedding_module.feat2dim[feature_name]][feature_name].values()
                    feature_id_jagged = feature_id_jagged.reshape(-1, multi)
                    mask = (feature_id_jagged != self.embedding_module.padding_index).float().unsqueeze(-1)
                    emb_masked = feature_value * mask  # 去除0值的值
                    sum_over_mask = emb_masked.sum(dim=1)
                    valid_count = mask.sum(dim=1)
                    feature_value = sum_over_mask / (valid_count + 1e-8)

            feature_emb_list.append(feature_value)

        feature_embs = torch.cat(feature_emb_list, dim=-1)
        # candidate_ids = model_input.get('candidate_ids')
        # candidate_lengths = model_input.get("candidate_lengths")
        # candidate_offsets = torch.cat([torch.tensor([0], device=('npu')),candidate_lengths])
        # candidate_offsets = candidate_offsets.cumsum(dim=0)
        # feature_embs = jagged_to_padded_dense(values=feature_embs,
        #                                       offsets=candidate_offsets,
        #                                       max_length=candidate_ids.shape[1]) # [B, C, D]
        return feature_embs

    def _get_feature_values(self, all_features: Dict[str, torch.Tensor], feature_names) -> ChainMap[str, torch.Tensor]:
        jt_dicts: List[Dict[str, JaggedTensor]] = []
        for ec_attr in feature_names.keys():
            jt_dicts.append(getattr(self, ec_attr)(all_features[ec_attr]).wait())
        embs_dict = ChainMap(*jt_dicts)
        return embs_dict

    def _init_can_embs(self, model_input: Dict[str, torch.Tensor]):
        self.feature_values_can_multi_cache = self._get_feature_values(model_input, self.dim2feats_can_multi)
        self.feature_values_can_single_cache = self._get_feature_values(model_input, self.dim2feats_can_single)

    def _reset_can_embs(self):
        if self.feature_values_can_multi_cache is not None:
            del self.feature_values_can_multi_cache
            self.feature_values_can_multi_cache = None
        if self.feature_values_can_single_cache is not None:
            del self.feature_values_can_single_cache
            self.feature_values_can_single_cache = None

    def prepare_can_embeddings_if_necessary(self, model_input: Dict[str, torch.Tensor]):
        return initializing(functools.partial(self._init_can_embs, model_input), self._reset_can_embs)

    def _get_can_embeddings(self, model_input: Dict[str, torch.Tensor], feature_names: list[str]) -> torch.Tensor:
        '''
        model_input: Dict of [B, L] or [B, L, M] or [B, C] or [B, C, L] or [B, C, M] or [B]
        '''
        # can_model_input = self._prepare_can_jagged_batch(model_input)

        emb_list_single = []
        emb_list_multi = []
        with self.prepare_can_embeddings_if_necessary(model_input):
            for feature_name in feature_names:
                feature_id = model_input[feature_name]
                feature_dtype = self.embedding_module._feature_dtypes.get(feature_name)
                if feature_dtype == "con":  # TODO: continous
                    raise ValueError('continous features are not allowed in CAN.')
                elif feature_dtype == "multi":
                    multi_emb = self.feature_values_can_multi_cache[feature_name].values()  # [N * L, D]
                    multi = feature_id.size(-1)
                    D = multi_emb.size(-1)
                    multi_emb = multi_emb.reshape(-1, multi, D)  # [N, L, D]
                    feature_id_jagged = model_input[self.feat2dim_can_multi[feature_name]][feature_name].values()
                    feature_id_jagged = feature_id_jagged.reshape(-1, multi)  # [N, L]
                    mask = (feature_id_jagged != self.embedding_module.padding_index).float().unsqueeze(-1)  # [N, L, 1]
                    multi_emb = multi_emb * mask  # [N, L, D]
                    emb_list_multi.append(multi_emb)
                elif feature_dtype == "int":
                    single_emb = self.feature_values_can_single_cache[feature_name].values()  # [N, 16*8*4]
                    emb_list_single.append(single_emb)
                else:
                    pass
            multi_embs = torch.cat(emb_list_multi, dim=-2)  # [N, L_t, D]
            single_embs = torch.stack(emb_list_single, dim=-2)  # [N, N_s, 16*8*4]

            # CAN
            out_seq, cur_idx = [], 0
            can_all_layers = [self.can_input_dim] + self.can_hidden_dims
            hidden_out = multi_embs
            for i, (in_dim, out_dim) in enumerate(zip(can_all_layers[:-1], can_all_layers[1:])):
                mlp_w = single_embs[..., cur_idx: cur_idx + in_dim * out_dim]  # [N, N_s, in_dim*out_dim]
                cur_idx = cur_idx + in_dim * out_dim
                mlp_w = mlp_w.reshape(mlp_w.shape[:-1] + (in_dim, out_dim))  # [N, N_s, in_dim, out_dim]
                if i == 0:
                    hidden_out = torch.einsum('aik,ajkl->aijl', hidden_out, mlp_w)  # [N, L_t, N_s, out_dim]
                else:
                    hidden_out = torch.einsum('aijk,ajkl->aijl', hidden_out, mlp_w)  # [N, L_t, N_s, out_dim]
                hidden_out = torch.tanh(hidden_out)
                out_seq.append(hidden_out)

            out_seq = torch.cat(out_seq, dim=-1)  # [N, L_t, N_s, D]
            out_seq = out_seq.reshape(out_seq.shape[0], -1)  # [N, D]
            # candidate_ids = model_input.get('candidate_ids')
            # candidate_lengths = model_input.get("candidate_lengths")
            # candidate_offsets = torch.cat([torch.tensor([0], device=('npu')),candidate_lengths])
            # candidate_offsets = candidate_offsets.cumsum(dim=0)
            # out_seq = jagged_to_padded_dense(values=out_seq,
            #                                  offsets=candidate_offsets,
            #                                  max_length=candidate_ids.shape[1]) # [B, C, D]
        return out_seq

    def forward(self,
                model_input: Dict[str, torch.Tensor],
                seq_embeddings: torch.Tensor):
        '''
        model_input: Dict
        seq_embeddings: torch.Tensor, [B, D]
        '''
        candidate_lengths = model_input.get("candidate_lengths")
        candidate_ids = model_input.get("candidate_ids")
        candidate_offsets = torch.cat([torch.tensor([0], device=('npu')), candidate_lengths])
        candidate_offsets = candidate_offsets.cumsum(dim=0)
        if not self.enable_jagged_ops:
            seq_embeddings = dense_to_jagged(dense=seq_embeddings,
                                             offsets=candidate_offsets,
                                             max_length=candidate_ids.shape[1]
                                             )
        feature_names = self.feature_groups['candidate_dlrm']['features']
        dnn_feature_embs = []
        cross_feature_embs = []
        ppnet_feature_embs = []
        final_feature_embs = []

        cand_feature_embs = self._get_feature_embeddings(model_input, feature_names)

        if self.use_dlrm_dnn:
            dnn_feature_embs.append(cand_feature_embs)
        if self.use_dlrm_cross:
            cross_feature_embs.append(cand_feature_embs)

        if self.use_dlrm_ppnet:
            ppnet_feature_embs.append(cand_feature_embs)

        if self.seq_as_cross_input and self.use_dlrm_cross:
            cross_feature_embs.append(seq_embeddings)

        if self.seq_as_dnn_input and self.use_dlrm_dnn:
            dnn_feature_embs.append(seq_embeddings)
        if self.seq_as_ppnet_input and self.use_dlrm_ppnet:
            ppnet_feature_embs.append(seq_embeddings)
        if self.seq_as_final_input:
            final_feature_embs.append(seq_embeddings)

        # CAN
        if self.use_dlrm_can:
            can_feature_embs = self._get_can_embeddings(model_input, self.can_selected_features)  # [32, 400, 34380]
            if self.can_as_dnn_input and self.use_dlrm_dnn:
                dnn_feature_embs.append(can_feature_embs)
            if self.can_as_dcn_input and self.use_dlrm_cross:
                cross_feature_embs.append(can_feature_embs)
            if self.can_as_final_input:
                final_feature_embs.append(can_feature_embs)

        # Cross
        if self.use_dlrm_cross:
            cross_feature_embs = torch.concat(cross_feature_embs, dim=-1)
            cross_feature_embs = self.cross(cross_feature_embs)
            final_feature_embs.append(cross_feature_embs)

        # DNN
        if self.use_dlrm_dnn:
            dnn_feature_embs = torch.concat(dnn_feature_embs, dim=-1)
            dnn_feature_embs = self.dnn(dnn_feature_embs)
            final_feature_embs.append(dnn_feature_embs)

        # PPNet
        if self.use_dlrm_ppnet:
            ppnet_feature_embs = torch.concat(ppnet_feature_embs, dim=-1)
            ppnet_gate_embs = self._get_feature_embeddings(model_input, self.ppnet_gate_features)  # [B, C, D]
            ppnet_feature_embs = self.ppnet([ppnet_feature_embs, ppnet_gate_embs])
            final_feature_embs.append(ppnet_feature_embs)

        # Linear
        if self.use_dlrm_linear:  # TODO: implementation is different from the baseline in TF
            linear_feature_embs = self.dlrm_linear(cand_feature_embs)  # [B, 1]
            final_feature_embs.append(linear_feature_embs)
        final_feature_embs = torch.concat(final_feature_embs, dim=-1)
        # final_feature_embs = jagged_to_padded_dense(values=final_feature_embs,
        #                                             offsets=candidate_offsets,
        #                                             max_length=candidate_ids.shape[1])
        return final_feature_embs


@ModelRegistry.register(
    req_subs={"EmbeddingModule", "InputFeaturesPreprocessorModule", "SequentialModule", "AttentionMaskModule",
              "FeedForwardModule", "OutputPostprocessorModule", "LossModule"}, opt_subs={"NegativesSampler"})
class MTGR_model_v2(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        feat_conf = common_hp["feature_conf"]
        model_conf = common_hp["model_conf"]
        self.history_length = common_hp['data_loader_conf'].get('history_length', 400)
        self.num_rerank = common_hp['data_loader_conf'].get('num_rerank', 400)
        self.enable_jagged_ops = model_conf.get("enable_jagged_ops", False) and self.training

        # feat_conf
        self.feature_groups = feat_conf.get("feature_groups")
        self.hist_ts_key = feat_conf.get("history_timestamps_column", FeatConst.DFLT_HIST_TS_KEY)
        self.hist_items_key = feat_conf.get("history_items_key", FeatConst.DFLT_HIST_ITEM_KEY)
        self.hist_ratings_key = feat_conf.get("history_ratings_column", FeatConst.DFLT_HIST_RATINGS_KEY)
        self.cand_ts_key = feat_conf.get("candidate_timestamps_column", FeatConst.DFLT_CAND_TS_KEY)
        self.cand_items_key = feat_conf.get("candidate_items_key", FeatConst.DFLT_CAND_ITEM_KEY)
        self.cand_ratings_key = feat_conf.get("candidate_ratings_column", FeatConst.DFLT_CAND_RATINGS_KEY)
        fuse_ia = feat_conf.get("fuse_ia", False)
        self.token_per_item = 2 if not fuse_ia else 1

        # model_hp
        model_hp = model_cfg[Const.HP]
        self._verbose = model_hp.get("verbose", True)
        self.concat_user_embeddings = self.model_cfg.get("use_user_embeddings_for_rerank", False)

        # embedding tables
        self.embedding_module: EmbeddingModule = self.init_sub_model("EmbeddingModule")

        # sequential models
        if model_hp.get('use_seq_model', True):
            self.input_propcessor_module: InputFeaturesPreprocessorModule = self.init_sub_model(
                "InputFeaturesPreprocessorModule")
            self.sequence_model: SequentialModule = self.init_sub_model("SequentialModule")
            self.attention_mask_module: AttentionMaskModule = self.init_sub_model("AttentionMaskModule")
        else:
            assert model_cfg['sub_models']['DLRMModule'][Const.HP]['seq_as_cross_input'] == False and \
                   model_cfg['sub_models']['DLRMModule'][Const.HP]['seq_as_dnn_input'] == False and \
                   model_cfg['sub_models']['DLRMModule'][Const.HP]['seq_as_ppnet_input'] == False and \
                   model_cfg['sub_models']['DLRMModule'][Const.HP]['seq_as_final_input'] == False

        # DLRM
        self.use_dlrm = True
        if model_cfg['sub_models'].get('DLRMModule'):
            self.use_dlrm = model_cfg['sub_models']['DLRMModule'][Const.HP].get('use_dlrm', True)
            model_cfg['sub_models']['DLRMModule']['embedding_module'] = self.embedding_module
            model_cfg['sub_models']['DLRMModule']['item_embedding_dim'] = model_conf['item_embedding_dim']
            self.dlrm_model: DLRM_model_v2 = self.init_sub_model('DLRMModule')

        if 'RankMixerModule' in self.model_cfg['sub_models']:
            x_input_dim = 0
            if model_cfg['sub_models'].get('DLRMModule'):
                x_input_dim += self.dlrm_model.final_input_dim
            else:
                x_input_dim += model_conf['item_embedding_dim']
            if model_cfg['sub_models']['RankMixerModule'][Const.HP].get('use_candidate', True):
                for i, j in self.embedding_module._feature_dims.items():
                    if i in self.feature_groups.get('candidate').get('features'):
                        x_input_dim += j
            if model_cfg['sub_models']['RankMixerModule'][Const.HP].get('use_seq', True):
                x_input_dim += model_conf['item_embedding_dim']
            self.model_cfg['sub_models']['RankMixerModule']['hp']['x_input_dim'] = x_input_dim
            self.rankmixer = self.init_sub_model('RankMixerModule')

        # SpatialTemporalModule
        if model_cfg['sub_models']['SequentialModule']['name'] == "WideDeepModule" or \
                model_cfg['sub_models']['SequentialModule']['name'] == "SpatialTemporalModule":
            self.wide_feature_groups = model_cfg['sub_models']['SequentialModule']['wide_feature_groups']
        else:
            self.wide_feature_groups = None

        # negative sampler
        self.negative_sampler: NegativesSampler = None if "NegativesSampler" not in model_cfg[Const.SUB_MODELS] \
            else self.init_sub_model("NegativesSampler")
        if self.negative_sampler is not None:
            self.negative_sampler.load_embedding_module(self.embedding_module)
        self._max_sequence_length: int = self.history_length + self.num_rerank

        # predictor
        if 'LinearModuleForRerankScore' in self.model_cfg['sub_models']['FeedForwardModule']['sub_models'] and \
                model_cfg['sub_models'].get('DLRMModule'):
            self.model_cfg['sub_models']['FeedForwardModule']['sub_models'] \
                ['LinearModuleForRerankScore']['hp']['input_dim'] = self.dlrm_model.final_input_dim
        else:
            self.model_cfg['sub_models']['FeedForwardModule']['sub_models'] \
                ['LinearModuleForRerankScore']['hp']['input_dim'] = model_conf['item_embedding_dim']

        if 'RankMixerModule' in self.model_cfg['sub_models']:
            self.model_cfg['sub_models']['FeedForwardModule']['sub_models'] \
                ['LinearModuleForRerankScore']['hp']['input_dim'] = self.rankmixer._embedding_dim

        self.feed_forward_module: FeedForwardModule = self.init_sub_model("FeedForwardModule")

        # loss
        self.output_processor_module: OutputPostprocessorModule = self.init_sub_model("OutputPostprocessorModule")
        self.loss_module: LossModule = self.init_sub_model("LossModule")

        self.reset_params()

    def reset_params(self):
        for name, params in self.named_parameters():
            if ("sequence_model" in name) or ("embedding_module" in name) or ('dlrm_model' in name):
                if self._verbose:
                    logging.info("Skipping init for %s", name)
                continue
            try:
                torch.nn.init.xavier_normal_(params.data)
                if self._verbose:
                    logging.info("Initialize %s as xavier normal: %s params", name, params.data.shape[0])
            except Exception:
                if self._verbose:
                    logging.info("Failed to initialize %s: %s params", name, params.data.shape[0])

    @property
    def embedding_type(self):
        return self.embedding_module.embedding_type

    def get_embeddings(self, model_inputs):

        past_embeddings = self.embedding_module.get_ui_embeddings(
            group_name=FeatConst.HIST_PFX, input_features=model_inputs
        )
        candidate_embeddings = self.embedding_module.get_ui_embeddings(
            group_name=FeatConst.CAND_PFX, input_features=model_inputs
        )
        user_feature_embs = self.embedding_module.get_ui_embeddings(
            group_name=FeatConst.USER_PFX, input_features=model_inputs
        )
        feature_values_cache = self.embedding_module.feature_values_cache

        return past_embeddings, user_feature_embs, candidate_embeddings, feature_values_cache

    def get_single_feature_embeddings(self, model_inputs, history_feature_group, candidate_feature_group):
        # history_embeddings = self.embedding_module._get_feature_embeddings(model_inputs, [history_feature_name])
        # candidate_embeddings = self.embedding_module._get_feature_embeddings(model_inputs, [candidate_feature_name])
        history_embeddings = self.embedding_module.get_ui_embeddings(
            group_name=history_feature_group, input_features=model_inputs
        )
        candidate_embeddings = self.embedding_module.get_ui_embeddings(
            group_name=candidate_feature_group, input_features=model_inputs
        )
        return history_embeddings, candidate_embeddings

    def generate_input_sequence(
            self,
            model_inputs: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        综合序列信息，生成user, item1, action1, item2, action2...形式的输入序列。

        :return user_embeddings: 拼接后的输入给模型的token序列，形如user, item1, action1, item2, action2...
        :return past_embeddings: 原始的商品token序列，形如item1, item2, item3, ...
        :return all_timestamps: 时间戳序列
        """
        past_embeddings, user_feature_embs, candidate_embeddings, feature_values_cache = self.get_embeddings(
            model_inputs)
        # 如果是在推理时，将历史序列和候选集序列拼接后返回；如果在训练，则只返回历史序列。

        candidate_lengths = None if self.embedding_type == EmbeddingType.LOCAL \
            else model_inputs.get("candidate_lengths")
        past_lengths_after_input_processor, user_embeddings, numrerank_mask, x_offsets = self.input_propcessor_module(
            history_embeddings=past_embeddings,
            candidate_embeddings=candidate_embeddings,
            history_lengths=model_inputs.get('history_lengths'),
            candidate_lengths=candidate_lengths,
            history_ids=model_inputs.get('history_ids'),
            candidate_ids=model_inputs.get('candidate_ids'),
            user_feature_embs=user_feature_embs,
            history_ratings=model_inputs.get(self.hist_ratings_key),
            candidate_ratings=model_inputs.get(self.cand_ratings_key),
            enable_jagged_ops=self.enable_jagged_ops
        )
        all_timestamps = torch.concat(
            [model_inputs.get(self.hist_ts_key), model_inputs.get(self.cand_ts_key)], dim=1)

        if self.wide_feature_groups is not None:
            # wide_feauture_names：[['history_app_second_type', 'candidate_app_second_type'],....]
            wide_embeddings = []
            for history_feature_group, candidate_feature_group in self.wide_feature_groups:
                history_single_embeddings, candidate_single_embeddings = self.get_single_feature_embeddings(
                    model_inputs, history_feature_group, candidate_feature_group)
                _, a_embedding, numrerank_mask, x_offsets = self.input_propcessor_module(
                    history_embeddings=history_single_embeddings,
                    candidate_embeddings=candidate_single_embeddings,
                    # history_lengths=torch.ones((past_embeddings.shape[0],), device=past_embeddings.device) * 400,
                    history_lengths=model_inputs.get("history_lengths"),
                    candidate_lengths=candidate_lengths,
                    history_ids=model_inputs.get('history_ids'),
                    candidate_ids=model_inputs.get('candidate_ids'),
                    user_feature_embs=user_feature_embs,
                    history_ratings=model_inputs.get(self.hist_ratings_key),
                    candidate_ratings=model_inputs.get(self.cand_ratings_key),
                    enable_jagged_ops=self.enable_jagged_ops
                )
                wide_embeddings.append(a_embedding)
            return user_embeddings, all_timestamps, past_lengths_after_input_processor, feature_values_cache, wide_embeddings, numrerank_mask, x_offsets
        return user_embeddings, all_timestamps, past_lengths_after_input_processor, feature_values_cache, numrerank_mask, x_offsets

    def generate_user_embeddings(
            self,
            past_lengths: torch.Tensor,
            all_timestamps: torch.Tensor,
            seq_embeddings: torch.Tensor,
            attn_mask: torch.Tensor,
            numrerank_mask: torch.Tensor,
            x_offsets: torch.Tensor,
            cache: Optional[List[TransformerCacheState]] = None,
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            return_cache_states: bool = False,
            num_rerank: int = 0
    ) -> torch.Tensor:
        """
        综合序列信息，生成用户 embedding.
        [B, N] -> [B, N, D].
        """
        item_embeddings, _ = self.sequence_model(
            x=seq_embeddings,
            x_offsets=x_offsets,
            all_timestamps=all_timestamps,
            attn_mask=attn_mask,
            past_lengths=past_lengths,
            num_rerank=num_rerank,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )
        # 只返回候选集的部分的输出的商品token
        if self.enable_jagged_ops:
            _item_embeddings = item_embeddings[numrerank_mask]
        else:
            _item_embeddings = item_embeddings[:, -num_rerank:, :]

        if self.concat_user_embeddings:
            user_embeddings = item_embeddings[:, :1, :].repeat(1, _item_embeddings.shape[1], 1)
            item_embeddings = torch.cat([user_embeddings, _item_embeddings], dim=-1)
        else:
            item_embeddings = _item_embeddings

        return self.output_processor_module(item_embeddings)

    def generate_user_embeddings_wideDeep(
            self,
            past_lengths: torch.Tensor,
            all_timestamps: torch.Tensor,
            seq_embeddings: torch.Tensor,
            attn_mask: torch.Tensor,
            numrerank_mask: torch.Tensor,
            x_offsets: torch.Tensor,
            cache: Optional[List[TransformerCacheState]] = None,
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            return_cache_states: bool = False,
            num_rerank: int = 0,
            wide_embeddings=None
    ) -> torch.Tensor:
        """
        综合序列信息，生成用户 embedding.
        [B, N] -> [B, N, D].
        """
        item_embeddings, _ = self.sequence_model(
            x=seq_embeddings,
            x_offsets=x_offsets,
            all_timestamps=all_timestamps,
            attn_mask=attn_mask,
            past_lengths=past_lengths,
            num_rerank=num_rerank,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
            wide_embeddings=wide_embeddings
        )
        # 如果推理时，只返回候选集的部分的输出的商品token；如果训练时，返回所有商品的输出的token。
        if self.enable_jagged_ops:
            _item_embeddings = item_embeddings[numrerank_mask]
        else:
            _item_embeddings = item_embeddings[:, -num_rerank:, :]

        if self.concat_user_embeddings:
            user_embeddings = item_embeddings[:, :1, :].repeat(1, _item_embeddings.shape[1], 1)
            item_embeddings = torch.cat([user_embeddings, _item_embeddings], dim=-1)
        else:
            item_embeddings = _item_embeddings

        return self.output_processor_module(item_embeddings)

    def _model_input_preprocess(self, model_input: dict):
        # model_input class : <class 'torch.fx.proxy.Proxy'>
        return model_input

    def forward(
            self,
            model_input: dict
    ) -> torch.Tensor | dict:
        """
        生成式推荐大模型前向传播过程

        :param model_input: 传入的字典，里面包括商品的特征id序列，用户的特征和其他序列信息。
        """
        # if int(os.getenv("RANK")) == 0:
        #     logging.info(f'model_input history ts: {model_input[self.hist_ts_key][0]}')

        # num_rerank: 训练/候选集商品的数量
        model_input = self._model_input_preprocess(model_input)

        num_rerank = model_input.get('candidate_ids').shape[1]
        with self.embedding_module.prepare_embeddings_if_necessary(model_input):
            if self.model_cfg[Const.HP].get('use_seq_model', True):
                attn_mask = self.attention_mask_module(model_input, self.history_length, num_rerank,
                                                       self.enable_jagged_ops)
                if self.wide_feature_groups is None:
                    seq_embeddings, all_timestamps, past_lengths_after_input_processor, feature_values_cache, numrerank_mask, x_offsets = \
                        self.generate_input_sequence(model_inputs=model_input)
                    encoded_embeddings = self.generate_user_embeddings(past_lengths=past_lengths_after_input_processor,
                                                                       all_timestamps=all_timestamps,
                                                                       seq_embeddings=seq_embeddings,
                                                                       attn_mask=attn_mask,
                                                                       numrerank_mask=numrerank_mask,
                                                                       x_offsets=x_offsets,
                                                                       num_rerank=num_rerank)
                else:
                    seq_embeddings, all_timestamps, past_lengths_after_input_processor, feature_values_cache, wide_embeddings, numrerank_mask, x_offsets = \
                        self.generate_input_sequence(model_inputs=model_input)
                    encoded_embeddings = self.generate_user_embeddings_wideDeep(
                        past_lengths=past_lengths_after_input_processor,
                        all_timestamps=all_timestamps,
                        seq_embeddings=seq_embeddings,
                        attn_mask=attn_mask,
                        numrerank_mask=numrerank_mask,
                        x_offsets=x_offsets,
                        num_rerank=num_rerank,
                        wide_embeddings=wide_embeddings)
            else:
                seq_embeddings = None
                encoded_embeddings = None

            if self.use_dlrm and self.model_cfg['sub_models'].get('DLRMModule'):
                with record_function("## DLRMmoudle ##"):
                    encoded_embeddings = self.dlrm_model(model_input, encoded_embeddings)

            candidate_lengths = model_input.get("candidate_lengths")
            candidate_ids = model_input.get("candidate_ids").shape[1]
            candidate_offsets = torch.cat([torch.tensor([0], device=('npu')), candidate_lengths])
            candidate_offsets = candidate_offsets.cumsum(dim=0)

            if 'RankMixerModule' in self.model_cfg['sub_models'] and (not self.enable_jagged_ops):
                rankmixer_emb_list = []
                if self.model_cfg['sub_models'].get('DLRMModule'):
                    encoded_embeddings = jagged_to_padded_dense(values=encoded_embeddings,
                                                                offsets=candidate_offsets,
                                                                max_length=candidate_ids)
                # print("encoded_embeddings_dlrm:", encoded_embeddings.shape)
                rankmixer_emb_list.append(encoded_embeddings)
                # with record_function("## RankMixerModule ##"):
                if self.rankmixer.model_cfg[Const.HP].get('use_candidate', True):
                    candidate_feature_embs, _ = self.embedding_module._get_feature_embeddings_dwt(
                        model_input,
                        self.feature_groups.get('candidate').get("features")
                    )
                    candidate_feature_embs = jagged_to_padded_dense(values=candidate_feature_embs,
                                                                    offsets=candidate_offsets,
                                                                    max_length=candidate_ids)
                    # print("candidate_feature_embs:", candidate_feature_embs.shape)
                    rankmixer_emb_list.append(candidate_feature_embs)
                if self.rankmixer.model_cfg[Const.HP].get('use_seq', True):
                    num_rerank = model_input.get('candidate_ids').shape[1]
                    seq_cand = seq_embeddings[:, -num_rerank:, :]
                    rankmixer_emb_list.append(seq_cand)
                rankmixer_emb = torch.cat(rankmixer_emb_list, dim=-1)
                # print("rankmixer_emb:", rankmixer_emb.shape)
                encoded_embeddings, l1_loss, sparsity = self.rankmixer(rankmixer_emb, self.enable_jagged_ops)
                # print("encoded_embeddings_final:", encoded_embeddings.shape)
                # print("rankmixer_emb_final:", rankmixer_emb.shape)
            elif self.enable_jagged_ops and 'RankMixerModule' in self.model_cfg['sub_models']:
                rankmixer_emb_list = []
                # print("encoded_embeddings_dlrm:", encoded_embeddings.shape)
                rankmixer_emb_list.append(encoded_embeddings)
                # with record_function("## RankMixerModule ##"):
                if self.rankmixer.model_cfg[Const.HP].get('use_candidate', True):
                    candidate_feature_embs, _ = self.embedding_module._get_feature_embeddings_dwt(
                        model_input,
                        self.feature_groups.get('candidate').get("features")
                    )
                    # print("candidate_feature_embs:", candidate_feature_embs.shape)
                    rankmixer_emb_list.append(candidate_feature_embs)
                if self.rankmixer.model_cfg[Const.HP].get('use_seq', True):
                    # print("seq_embeddings:", seq_embeddings.shape)
                    seq_cand = seq_embeddings[numrerank_mask]
                    rankmixer_emb_list.append(seq_cand)
                    # print("rankmixer_emb_list:", rankmixer_emb_list[0].shape)
                rankmixer_emb = torch.cat(rankmixer_emb_list, dim=-1)
                # print("rankmixer_emb:", rankmixer_emb.shape)
                encoded_embeddings, l1_loss, sparsity = self.rankmixer(rankmixer_emb, self.enable_jagged_ops)


        results = self.feed_forward_module(encoded_embeddings, candidate_offsets, candidate_ids)

        if not self.training or torch.onnx.is_in_onnx_export():
            if self.enable_jagged_ops:
                scores = jagged_to_padded_dense(values=results["rerank_score"].unsqueeze(-1),
                                                offsets=candidate_offsets,
                                                max_length=candidate_ids)
                results = {"rerank_score": scores}
                return results
            elif (self.use_dlrm and self.model_cfg['sub_models'].get('DLRMModule')) and 'RankMixerModule' not in \
                    self.model_cfg['sub_models']:
                scores = jagged_to_padded_dense(values=results["rerank_score"].unsqueeze(-1),
                                                offsets=candidate_offsets,
                                                max_length=candidate_ids)
                results = {"rerank_score": scores}
                # results = {"rerank_score": results["rerank_score"]}
                return results
            else:
                return {"rerank_score": results["rerank_score"].unsqueeze(-1)}
        else:
            loss = self.loss_module(past_embeddings=seq_embeddings,
                                    encoded_embeddings=encoded_embeddings,
                                    predictions=results,
                                    model_inputs=model_input,
                                    negative_sampler=self.negative_sampler,
                                    offsets=candidate_offsets,
                                    candi_length=candidate_ids)
            return loss
