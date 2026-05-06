import os
from dataclasses import dataclass
from typing import List, Union

import torch
import logging

from data.concat_dataset import DatasetMusicV1, DatasetAdsV1, DatasetV8, DatasetAG
from modeling.generic.utils.constants import Const, FeatConst


@dataclass
class RecoDataset:
    """
    数据集类，用于存储训练和评估数据集的相关信息。
    """
    max_sequence_length: int
    num_unique_items: int
    max_item_id: int
    all_item_ids: List[int]
    train_dataset: torch.utils.data.Dataset
    eval_dataset: torch.utils.data.Dataset


def get_format_csv(data_dir, pth) -> str:
    return os.path.join(data_dir, pth)


def get_reco_dataset(
        dataset: str,
        data_dir: str,
        max_sequence_length: int,
        rank: int,
        world_size: int,
        pth: Union[List[str], str],
        chronological: bool = True,
        feature_conf: dict = None,
        num_rerank=256,
        history_length=400,
        use_jagged_data: bool = False
) -> RecoDataset:
    """
    创建并返回一个RecoDataset对象, 包含训练和评估数据集。
    
    :param data_dir: 数据目录。
    :param max_sequence_length: 序列的最大长度。
    :param rank: 全局rank ID。
    :param world_size: 分布式并行总进程数。
    :param pth: 文件路径。
    :param chronological: 是否按时间正序排列。
    :param feature_conf: 特征配置字典。
    :return: RecoDataset对象, 包含训练和评估数据集的相关信息。
    """
    if dataset == "music-scalingraw-rank":
        train_dataset = DatasetMusicV1(
            ratings_file=get_format_csv(data_dir, pth),
            padding_length=max_sequence_length + 1,
            ignore_last_n=0,
            chronological=chronological,
            is_ads=False,
            sep=',',
            rank=rank,
            world_size=world_size,
            seq_columns=feature_conf.get('item_feature_columns'),
            seq_column_name=feature_conf.get('raw_item_info_column'),
            nonseq_columns=feature_conf.get('user_feature_columns'),
            userid_column_name=feature_conf.get('raw_userid_column'),
            itemid_column_name=feature_conf.get('raw_itemid_column'),
            ratings_column_name=feature_conf.get('raw_ratings_column'),
            timestamps_column_name=feature_conf.get('raw_timestamps_column'),
            padding_index=feature_conf.get('padding_index'),
            padding_length_pref=feature_conf.get('padding_length_pref'),
            is_train=True,
            negative_sample=feature_conf.get('negative_sample', False),
            cut_off_time=int(feature_conf.get('cut_off_time')),
            behavior_map=feature_conf.get('behavior_map')
        )
        eval_dataset = DatasetMusicV1(
            ratings_file=get_format_csv(data_dir, pth),
            padding_length=max_sequence_length,
            ignore_last_n=0,
            chronological=chronological,
            is_ads=False,
            sep=',',
            rank=rank,
            world_size=world_size,
            seq_columns=feature_conf.get('item_feature_columns'),
            seq_column_name=feature_conf.get('raw_item_info_column'),
            nonseq_columns=feature_conf.get('user_feature_columns'),
            userid_column_name=feature_conf.get('raw_userid_column'),
            itemid_column_name=feature_conf.get('raw_itemid_column'),
            ratings_column_name=feature_conf.get('raw_ratings_column'),
            timestamps_column_name=feature_conf.get('raw_timestamps_column'),
            padding_index=feature_conf.get('padding_index'),
            padding_length_pref=feature_conf.get('padding_length_pref'),
            is_train=False,
            negative_sample=feature_conf.get('negative_sample', False),
            cut_off_time=int(feature_conf.get('cut_off_time')),
            num_rerank=num_rerank,
            behavior_map=feature_conf.get('behavior_map')
        )
    elif dataset == "ads":
        train_dataset = DatasetAdsV1(
            ratings_file=get_format_csv(data_dir + '/out', pth),
            padding_length=max_sequence_length,
            ignore_last_n=0,
            chronological=chronological,
            time_desc=feature_conf.get('time_desc', False),
            sep=';',
            rank=rank,
            world_size=world_size,
            seq_columns=feature_conf.get('item_feature_columns'),
            nonseq_columns=feature_conf.get('user_feature_columns'),
            itemid_column_name=feature_conf.get('infer_items_key'),
            ratings_column_name=feature_conf.get('infer_ratings_key'),
            timestamps_column_name=feature_conf.get('infer_timestamps_key'),
            is_train=True,
            cut_off_time=int(feature_conf.get('cut_off_time')),
            use_repadding=feature_conf.get('use_repadding', False),
            sample_rate=feature_conf.get('train_sample_rate', 1.0),
            file_format='csv',
            split_chunks_by_worker_id=True,
        )
        eval_dataset = DatasetAdsV1(
            ratings_file=get_format_csv(data_dir + '/out', pth),
            padding_length=max_sequence_length - 1,
            ignore_last_n=0,
            chronological=chronological,
            time_desc=feature_conf.get('time_desc', False),
            sep=';',
            rank=rank,
            world_size=world_size,
            seq_columns=feature_conf.get('item_feature_columns'),
            nonseq_columns=feature_conf.get('user_feature_columns'),
            itemid_column_name=feature_conf.get('infer_items_key'),
            ratings_column_name=feature_conf.get('infer_ratings_key'),
            timestamps_column_name=feature_conf.get('infer_timestamps_key'),
            is_train=False,
            cut_off_time=int(feature_conf.get('cut_off_time')),
            num_rerank=num_rerank,
            sample_rate=feature_conf.get('test_sample_rate', 1.0),
            file_format='csv',
            split_chunks_by_worker_id=True,
        )

    elif dataset == "ag-rank":
        token_per_item = 1 if feature_conf.get("fuse_ia", True) else 2
        cut_off_time = feature_conf.get("cut_off_time", None)
        if cut_off_time is None:
            raise ValueError("cut_off_time in config should be set as date for AG.")
        # finetune 新增:微调下界(YYYYMMDD,含)。仅 train 分支生效;
        # eval 分支始终用单一 cut_off_time(只看 cut_off 之后的候选)。
        cut_off_time_lower = feature_conf.get("cut_off_time_lower", None)
        if isinstance(pth, str):
            train_pth = pth 
        elif isinstance(pth, list) and len(pth) == 2:
            train_pth = pth[0]
        train_dataset = DatasetAG(
            ratings_file=get_format_csv(data_dir, train_pth),
            ignore_last_n=0,
            chronological=chronological,
            is_ads=feature_conf.get('time_desc', True),
            sep=';',
            rank=rank,
            world_size=world_size,
            history_items_key = feature_conf.get("history_items_key"),
            candidate_items_key = feature_conf.get("candidate_items_key"),
            history_feature_columns=feature_conf.get('history_item_feature_columns'),
            candidate_feature_columns=feature_conf.get('candidate_item_feature_columns'),
            nonseq_columns=feature_conf.get('user_feature_columns'),
            history_ratings_column_name=feature_conf.get('history_ratings_column'),
            candidate_ratings_column_name=feature_conf.get('candidate_ratings_column'),
            history_timestamps_column_name=feature_conf.get('history_timestamps_column'),
            candidate_timestamps_column_name=feature_conf.get('candidate_timestamps_column'),
            history_date_column_name=feature_conf.get('history_date_column'),
            candidate_date_column_name=feature_conf.get('candidate_date_column'),
            is_train=True,
            cut_off_time=cut_off_time,
            cut_off_time_lower=cut_off_time_lower,    # ← finetune 新增
            history_length=history_length,
            num_rerank=num_rerank,
            token_per_item=token_per_item,
            use_jagged_data = use_jagged_data
        )

        if isinstance(pth, str):
            eval_pth = pth 
        elif isinstance(pth, list) and len(pth) == 2:
            eval_pth = pth[1]
        eval_dataset = DatasetAG(
            ratings_file=get_format_csv(data_dir, eval_pth),
            ignore_last_n=0,
            chronological=chronological,
            is_ads=feature_conf.get('time_desc', True),
            sep=';',
            rank=rank,
            world_size=world_size,
            history_items_key = feature_conf.get("history_items_key"),
            candidate_items_key = feature_conf.get("candidate_items_key"),
            history_feature_columns=feature_conf.get('history_item_feature_columns'),
            candidate_feature_columns=feature_conf.get('candidate_item_feature_columns'),
            nonseq_columns=feature_conf.get('user_feature_columns'),
            history_ratings_column_name=feature_conf.get('history_ratings_column'),
            candidate_ratings_column_name=feature_conf.get('candidate_ratings_column'),
            history_timestamps_column_name=feature_conf.get('history_timestamps_column'),
            candidate_timestamps_column_name=feature_conf.get('candidate_timestamps_column'),
            history_date_column_name=feature_conf.get('history_date_column'),
            candidate_date_column_name=feature_conf.get('candidate_date_column'),
            is_train=False,
            cut_off_time=cut_off_time,
            history_length=history_length,
            num_rerank=num_rerank,
            token_per_item=token_per_item
        )
    else:
        raise ValueError(f"Unsupported dataset {dataset}")

    all_item_ids = [x + 1 for x in range(Const.EXPECTED_NUM_UNIQUE_ITEMS)]
    max_item_id = Const.EXPECTED_NUM_UNIQUE_ITEMS

    return RecoDataset(
        max_sequence_length=max_sequence_length,
        num_unique_items=Const.EXPECTED_NUM_UNIQUE_ITEMS,
        max_item_id=max_item_id,
        all_item_ids=all_item_ids,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
