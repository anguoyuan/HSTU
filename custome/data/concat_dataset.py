import ast
import logging
import os
from typing import Dict, List, Optional, Tuple, Union
from itertools import compress
import random
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import torch
import traceback
from torch.utils.data import IterableDataset

from modeling.generic.utils.constants import DEFAULT_DATA_LEN, FeatConst


class MultiCSVIterator:
    """用于读取csv文件的迭代器。"""

    def __init__(self, file_paths: List[str], sep: str, rank: int, world_size: int, column_names, inner_delim,
                 itemid_column_name='item_id', file_format='csv', split_chunks_by_worker_id=True,
                 chunksize=50000) -> None:
        """
        初始化MultiCSVIterator。

        :param file_paths: csv文件路径列表。
        :param sep: csv文件的分隔符。
        :param rank: 用于分布式训练的rank，0表示主进程。
        :param world_size: 分布式并行总进程数。
        :param column_names: csv文件的列名。
        :param inner_delim: 内部分隔符，用于处理列中的多个值。
        :param itemid_column_name: 表示物品ID的列名，默认为'item_id'。
        """

        self.file_paths = file_paths
        self.sep = sep
        self.rank = rank
        self.world_size = world_size

        self.current_file_index = 0
        self.chunk_iterator = iter([])
        self.chunk_data = None
        self.current_data_index = 0
        self.total_read_count = 0

        self.column_names = column_names
        self.inner_delim = inner_delim

        self.itemid_column_name = itemid_column_name
        self.worker_info = torch.utils.data.get_worker_info()
        self.file_format = file_format
        self.split_chunks_by_worker_id = split_chunks_by_worker_id
        self.increment = 1 if self.split_chunks_by_worker_id else self.world_size
        self.chunk_size = chunksize

    def __iter__(self):
        return self

    def read_next_file(self):
        if self.current_file_index >= len(self.file_paths):
            logging.info('worker_id %s finished data iteration ', self.worker_info)
            raise StopIteration

        logging.info('worker_id %s : loading file_id %s : loading chunks from file: %s',
                     self.worker_info.id, self.current_file_index, self.file_paths[self.current_file_index])
        if self.file_format == 'orc':
            self.chunk_iterator = pd.read_orc(self.file_paths[self.current_file_index])
        elif self.file_format == 'csv':
            if self.split_chunks_by_worker_id:
                self.chunk_iterator = pd.read_csv(self.file_paths[self.current_file_index], sep=self.sep)
            else:
                self.chunk_iterator = pd.read_csv(self.file_paths[self.current_file_index], sep=self.sep,
                                                  chunksize=self.chunk_size)
        elif self.file_format == 'parquet':
            if self.split_chunks_by_worker_id:
                # Read entire file
                self.chunk_iterator = pd.read_parquet(self.file_paths[self.current_file_index], engine='pyarrow')
            else:
                # Read in chunks
                parquet_file = pq.ParquetFile(self.file_paths[self.current_file_index])
                self.chunk_iterator = (batch.to_pandas(types_mapper=pd.ArrowDtype) for batch in
                                    parquet_file.iter_batches(batch_size=self.chunk_size))
        self.current_file_index += 1

    def get_next_chunk(self):
        if self.split_chunks_by_worker_id:
            self.read_next_file()
            self.total_read_count = len(self.chunk_iterator) // self.world_size
            read_start_idx = self.total_read_count * self.rank
            self.chunk_data = self.chunk_iterator[read_start_idx: read_start_idx + self.total_read_count]
            del self.chunk_iterator
            self.current_data_index = 0
        else:
            try:
                self.chunk_data = self.chunk_iterator.__next__()
                self.total_read_count = len(self.chunk_data) - len(self.chunk_data) % self.world_size
                self.current_data_index = self.rank
            except StopIteration:
                # if get next chunk fails, read next file
                self.read_next_file()
                return self.__next__()

    def __next__(self):
        if self.chunk_data is None:
            self.get_next_chunk()

        if self.current_data_index < self.total_read_count:
            data = self.chunk_data.iloc[self.current_data_index]
            self.current_data_index += self.increment
            return data
        else:
            self.chunk_data = None
            return self.__next__()


class DatasetMusicV1(IterableDataset):
    """In chronological order."""

    def __init__(
            self,
            ratings_file: str,
            padding_length: int,
            ignore_last_n: int,
            shift_id_by: int = 0,
            chronological: bool = True,
            sample_ratio: float = 1.0,
            is_ads: bool = False,
            sep: str = ',',
            rank: int = 0,
            world_size: int = 1,
            seq_columns=None,
            seq_column_name='sequence_info',
            nonseq_columns=None,
            userid_column_name='aid',
            itemid_column_name='item_id',
            ratings_column_name='event_id',
            timestamps_column_name='time_stamp',
            inner_delim: str = ',',
            multi_value_delim: str = '#',
            continous_value_prefix: str = "",
            padding_index: int = 1,
            padding_length_pref: int = 10,
            is_train: bool = True,
            negative_sample=False,
            cut_off_time: int = 1733673600,
            num_rerank=200,
            behavior_map=None,
            file_format='orc',
            split_chunks_by_worker_id=True
    ) -> None:
        super().__init__()
        """
        初始化DatasetV8。

        :param ratings_file: 文件路径。
        :param padding_length: 填充长度。
        :param ignore_last_n: 忽略最后n个数据项。
        :param shift_id_by: ID偏移量。
        :param chronological: 默认为True, 处理后按时间正序排列。
        :param sample_ratio: 采样比例。
        :param is_ads: 是否为广告数据。
        :param sep: 分隔符。
        :param rank: 用于分布式训练的rank。
        :param world_size: 用于分布式训练的世界大小。
        :param seq_columns: 序列列名，这里指货/场特征名。
        :param nonseq_columns: 非序列列名，这里指人特征名。
        :param itemid_column_name: 物品ID列名。
        :param ratings_column_name: 评分列名。
        :param timestamps_column_name: 时间戳列名。
        :param inner_delim: 内部分隔符。
        :param is_train: 是否为训练数据。
        :param negative_sample: 控制负采样
        :param cut_off_time: 训练/验证切分时间戳, 默认为093000对应的时间戳。
        :param num_rerank: 离线评估时候选集大小
        :param behavior_map: 行为的正负样本映射
        :parma file_format: 
        """

        if seq_columns is None:
            raise ValueError('seq_columns should not be None')
        if nonseq_columns is None:
            raise ValueError('nonseq_columns should not be None')

        self.rank = rank
        self.world_size = world_size

        files = os.listdir(ratings_file)
        self.files = [ratings_file + '/' + f for f in files if f.startswith("part")]

        self.current_ratings_frame_len = 0
        self.current_ratings_frame_idx = 0
        self.data_idx = 0
        self.ratings_frame = None

        self._padding_length: int = padding_length
        self._ignore_last_n: int = ignore_last_n
        self._cache = dict()
        self._shift_id_by: int = shift_id_by
        self._chronological: bool = chronological
        self._sample_ratio: float = sample_ratio

        self._max_candidate_num: int = num_rerank
        self.seq_columns = seq_columns
        self.seq_column_name = seq_column_name
        self.nonseq_columns = nonseq_columns
        self.seq_feat_names = [k for k, _ in seq_columns.items()]
        self.num_seq_feat = len(self.seq_feat_names)
        self.nonseq_feat_names = [k for k, _ in nonseq_columns.items()]
        self.num_nonseq_feat = len(self.nonseq_feat_names)
        self.userid_column_name = userid_column_name
        self.itemid_column_name = itemid_column_name
        self.ratings_column_name = ratings_column_name
        self.timestamps_column_name = timestamps_column_name
        self.column_names = [ratings_column_name, timestamps_column_name]

        self.inner_delim = inner_delim
        self.multi_delim = multi_value_delim
        self.con_prefix = continous_value_prefix
        self.padding_index = padding_index
        self.padding_length_pref = padding_length_pref
        self.cut_off_time = cut_off_time
        self.is_train = is_train
        self.negative_sample = negative_sample
        self.behavior_map = behavior_map

        self.is_ads = is_ads
        self.sep = sep
        self.multi_csv_iterator = None
        self.file_format = file_format
        self.split_chunks_by_worker_id = split_chunks_by_worker_id

    def init_multi_csv_iterator(self) -> None:
        if not self.multi_csv_iterator:
            worker_info = torch.utils.data.get_worker_info()
            self.multi_csv_iterator = MultiCSVIterator(self.files, self.sep, self.rank, self.world_size,
                                                       self.column_names, self.inner_delim,
                                                       itemid_column_name=self.itemid_column_name,
                                                       file_format=self.file_format,
                                                       split_chunks_by_worker_id=self.split_chunks_by_worker_id)

    def __len__(self):
        # deprecated,不要用，这个不准确
        return DEFAULT_DATA_LEN

    def __iter__(self):
        # 当前df的数据索引
        it = map(self.load_item, self.multi_csv_iterator)
        return it

    def load_item(self, data) -> Dict[str, torch.Tensor]:
        """
        加载单个数据项。

        :param data: 单个数据项。
        :return: 处理后的数据项，包含 history_lengths, test_position, ratings, loss_weights, labels, timestamps
        """

        def eval_as_list(x: str, ignore_last_n) -> List[int]:
            x = x.replace(self.inner_delim, ',')
            y = ast.literal_eval(x)
            y_list = [y] if isinstance(y, int) else list(y)
            if ignore_last_n > 0:
                # for training data creation
                y_list = y_list[:-ignore_last_n]
            return y_list

        def eval_int_list(x, ignore_last_n: int, shift_id_by: int, sampling_kept_mask: Optional[List[bool]]) -> Tuple[
            List[int], int]:
            y = eval_as_list(x, ignore_last_n=ignore_last_n)
            if sampling_kept_mask is not None:
                y = [x for x, kept in zip(y, sampling_kept_mask) if kept]
            y_len = len(y)

            if shift_id_by > 0:
                y = [x + shift_id_by for x in y]
            return y, y_len

        def eval_complex_feat_list(x: str, x_dtype: List, padding_length_pref: int = 10):
            x = x.split(self.inner_delim)
            y_list = [a.strip() for a in x]
            for idx, element in enumerate(y_list):
                if x_dtype[idx] == "pref":
                    pref_value = [int(a) for a in element.split(self.multi_delim)]
                    # 进行padding
                    perf_len = len(pref_value)
                    if perf_len < padding_length_pref:
                        # pref特征的缺省值为1
                        pref_value = pref_value + [self.padding_index] * (padding_length_pref - perf_len)
                    else:
                        pref_value = pref_value[-padding_length_pref:]
                    y_list[idx] = pref_value
                elif x_dtype[idx] == "con":
                    if self.con_prefix is not None:
                        y_list[idx] = float(element.replace(self.con_prefix, ""))
                    else:
                        y_list[idx] = float(element)
                else:
                    y_list[idx] = int(element)
            return y_list, len(y_list)

        if self._sample_ratio < 1.0:
            raw_length = len(eval_as_list(data[self.itemid_column_name], self._ignore_last_n))
            sampling_kept_mask = (torch.rand((raw_length,), dtype=torch.float32) < self._sample_ratio).tolist()
        else:
            sampling_kept_mask = None

        ratings, ratings_len = eval_int_list(
            data[self.ratings_column_name], self._ignore_last_n, 0, sampling_kept_mask=sampling_kept_mask
        )

        item_id_str = data[self.itemid_column_name]
        iid_list = [int(x) for x in item_id_str.split(self.inner_delim)]
        original_len_iid = len(iid_list)

        # 处理训练数据标签
        label_dict = self.behavior_map
        labels = [label_dict.get(r, 0) for r in ratings]

        # 随机保留负rating
        if self.negative_sample != False:
            if self.negative_sample > 1.0 or self.negative_sample < 0.0:
                raise ValueError("negative_sample should be False or float between 0 and 1!")
            zero_rating_indices = [i for i, x in enumerate(labels) if x == 0]
            half_zeros_count = round(len(zero_rating_indices) * float(self.negative_sample))
            if half_zeros_count >= 1:
                random.seed(2024)
                remove_indices = random.sample(zero_rating_indices, half_zeros_count)
                retain_indices_rating = [i for i in range(len(iid_list)) if i not in remove_indices]
            else:
                retain_indices_rating = [i for i in range(len(iid_list))]
            iid_list = [iid_list[i] for i in retain_indices_rating]

        # 特征处理
        non_seq_feature = {}
        seq_feature_list = {}
        seq_feature_len = {}
        candidate_feature_list = {}
        hist_seq_feature_list = {}

        all_seq_feat_dtype = [v["dtype"] for _, v in self.seq_columns.items()]
        all_seq_feat_dtype = all_seq_feat_dtype[1:]
        all_seq_feat_dtype = all_seq_feat_dtype * original_len_iid
        all_seq_feats, num_all_seq_feats = eval_complex_feat_list(
            data[self.seq_column_name], all_seq_feat_dtype
        )
        num_seq_feats_per_item = int(num_all_seq_feats / original_len_iid)
        if num_seq_feats_per_item != len(self.seq_feat_names) - 1:
            raise ValueError("Number of seqence (item) features %s is mismatched with the input config %s" % (
                num_seq_feats_per_item + 1, len(self.seq_feat_names)))

        seq_feat_idx = 0
        for column_name in self.seq_feat_names:
            if column_name == self.itemid_column_name:
                seq_feature = iid_list
                seq_history_len = len(iid_list)
            else:
                seq_feature = []
                for x in range(original_len_iid):
                    index = x * num_seq_feats_per_item + seq_feat_idx
                    seq_feature.append(all_seq_feats[index])
                if self.negative_sample != False:
                    seq_feature = [seq_feature[i] for i in retain_indices_rating]
                seq_history_len = len(seq_feature)
                seq_feat_idx += 1

            seq_feature_list[column_name] = seq_feature
            seq_feature_len[column_name] = seq_history_len

        all_nonseq_feat_dtype = ["int"] + [v["dtype"] for _, v in self.nonseq_columns.items()]
        split_data, _ = eval_complex_feat_list(data[self.userid_column_name], all_nonseq_feat_dtype,
                                               padding_length_pref=self.padding_length_pref)
        all_nonseq_feats = split_data[1:]
        for nonseq_feat_idx, column_name in enumerate(self.nonseq_feat_names):
            feat_val = all_nonseq_feats[nonseq_feat_idx]
            non_seq_feature[column_name] = feat_val
        if self.negative_sample != False:
            ratings = [ratings[i] for i in retain_indices_rating]
        ratings_len = len(ratings)
        timestamps, timestamps_len = eval_int_list(
            data[self.timestamps_column_name], self._ignore_last_n, 0, sampling_kept_mask=sampling_kept_mask
        )
        if self.negative_sample != False:
            timestamps = [timestamps[i] for i in retain_indices_rating]
        timestamps_len = len(timestamps)

        if timestamps_len != ratings_len:
            raise ValueError("timestamps len %s differs from ratings len %s." % (timestamps_len, ratings_len))

        def _truncate_or_pad_seq(y: List, y_dtype: str, target_len: int, chronological: bool, ) -> List[int]:
            y_len = len(y)
            if y_len < target_len:
                if y_dtype == "int":
                    y = y + [self.padding_index] * (target_len - y_len)
                elif y_dtype == "con":
                    y = y + [0.0] * (target_len - y_len)
            else:
                if not chronological:
                    y = y[:target_len]
                else:
                    y = y[-target_len:]
            if len(y) != target_len:
                raise ValueError
            return y

        def candidate_feature_interleave(candidate_feature: list):
            return [candidate_feature[i // 2] for i in range(len(candidate_feature) * 2)]

        max_seq_len = self._padding_length - 1
        max_candidate_num = self._max_candidate_num

        split_pos = min([i if ts > self.cut_off_time else max_seq_len + 100000000 for i, ts in enumerate(timestamps)])
        hist_ratings = ratings[:split_pos]
        hist_timestamps = timestamps[:split_pos]
        history_length = min(len(hist_timestamps), max_seq_len)
        if not self.is_train:
            candidate_timestamps = timestamps[split_pos:]
            candidate_ratings = ratings[split_pos:]
        hist_ratings = _truncate_or_pad_seq(hist_ratings, "int", max_seq_len, self._chronological)
        hist_timestamps = _truncate_or_pad_seq(hist_timestamps, "int", max_seq_len, self._chronological)

        for k, v in seq_feature_list.items():
            v = v[:split_pos]
            hist_seq_feature_list[k] = v
        if not self.is_train:
            for candidate_k, candidate_v in seq_feature_list.items():
                candidate_v = candidate_v[split_pos:]
                candidate_feature_list["candidate_" + candidate_k] = candidate_v
        for k, v in hist_seq_feature_list.items():
            v = _truncate_or_pad_seq(v, self.seq_columns[k]["dtype"], max_seq_len, self._chronological)
            hist_seq_feature_list[k] = v
        if not self.is_train:
            candidate_timestamps = _truncate_or_pad_seq(candidate_timestamps, "int", max_candidate_num,
                                                        self._chronological)
            candidate_ratings = _truncate_or_pad_seq(candidate_ratings, "int",
                                                     max_candidate_num, self._chronological)
            # RAB calculation during inference:
            # during evaluation candidate item and user seq is arranged in this manner 
            # to allow different timestamps for different candidate item to be calculated simultaneously:
            # user_item_1 user_action_1 user_item_2 user_action_2 ....
            # .. candidate_item_1 candidate_item_1 candidate_item_2 candidate_item_2 
            candidate_ratings = candidate_feature_interleave(candidate_ratings)
            candidate_timestamps = candidate_timestamps + candidate_timestamps
            for candidate_k, candidate_v in candidate_feature_list.items():
                candidate_v = _truncate_or_pad_seq(candidate_v,
                                                   self.seq_columns[candidate_k.replace("candidate_", "")]["dtype"],
                                                   max_candidate_num, self._chronological)
                candidate_feature_list[candidate_k] = candidate_feature_interleave(candidate_v)

        # 处理训练数据标签
        labels = [label_dict.get(r, 0) for r in hist_ratings]

        # 处理模型权重
        loss_weights = [1 if r in label_dict else 0 for r in hist_ratings]

        ret = {
            "uid": 0,
            "history_lengths": history_length,
            "test_position": split_pos,
            "ratings": torch.tensor(hist_ratings, dtype=torch.int64),
            "loss_weights": torch.tensor(loss_weights, dtype=torch.int64),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "timestamps": torch.tensor(hist_timestamps, dtype=torch.int64)
        }

        for k, v in non_seq_feature.items():
            ret[k] = torch.tensor(v, dtype=torch.int64)

        for k, v in hist_seq_feature_list.items():
            if self.seq_columns[k]["dtype"] == "con":
                ret[k] = torch.tensor(v, dtype=torch.float32)
            else:
                ret[k] = torch.tensor(v, dtype=torch.int64)

        if not self.is_train:
            ret["candidate_ratings"] = torch.tensor(candidate_ratings, dtype=torch.int64)
            candidate_labels = [label_dict.get(r, 0) for r in candidate_ratings]
            ret["candidate_labels"] = torch.tensor(candidate_labels, dtype=torch.int64)
            ret["candidate_timestamps"] = torch.tensor(candidate_timestamps, dtype=torch.int64)
            for k, v in candidate_feature_list.items():
                if self.seq_columns[k.replace("candidate_", "")]["dtype"] == "con":
                    ret[k] = torch.tensor(v, dtype=torch.float32)
                else:
                    ret[k] = torch.tensor(v, dtype=torch.int64)
        return ret


class DatasetAdsV1(IterableDataset):
    """用于处理按时间逆序排列的数据集。"""

    def __init__(
            self,
            ratings_file: str,
            padding_length: int,
            ignore_last_n: int,
            shift_id_by: int = 0,
            chronological: bool = True,
            sample_ratio: float = 1.0,
            time_desc: bool = False,
            sep: str = ',',
            rank: int = 0,
            world_size: int = 1,
            seq_columns=None,
            nonseq_columns=None,
            itemid_column_name='item_id',
            ratings_column_name='event_id',
            timestamps_column_name='time_stamp',
            inner_delim: str = '^',
            is_train: bool = True,
            cut_off_time: int = 1727625600,
            num_rerank=200,
            use_repadding: bool = False,
            file_format: str = 'csv',
            split_chunks_by_worker_id: bool = False,
            sample_rate=1.0
    ) -> None:
        super().__init__()
        """
        初始化DatasetV8。

        :param ratings_file: 文件路径。
        :param padding_length: 填充长度。
        :param ignore_last_n: 忽略最后n个数据项。
        :param shift_id_by: ID偏移量。
        :param chronological: 默认为True, 处理后按时间正序排列。
        :param sample_ratio: 采样比例。
        :param is_ads: 是否为广告数据。
        :param sep: 分隔符。
        :param rank: 用于分布式训练的rank。
        :param world_size: 用于分布式训练的世界大小。
        :param seq_columns: 序列列名，这里指货/场特征名。
        :param nonseq_columns: 非序列列名，这里指人特征名。
        :param itemid_column_name: 物品ID列名。
        :param ratings_column_name: 评分列名。
        :param timestamps_column_name: 时间戳列名。
        :param inner_delim: 内部分隔符。
        :param is_train: 是否为训练数据。
        :param cut_off_time: 训练/验证切分时间戳, 默认为093000对应的时间戳。
        """

        if seq_columns is None:
            raise ValueError('seq_columns should not be None')
        if nonseq_columns is None:
            raise ValueError('nonseq_columns should not be None')

        self.rank = rank
        self.world_size = world_size

        try:
            files = os.listdir(ratings_file)
        except FileNotFoundError as fnf_error:
            logging.error('No such file or directory: %s', ratings_file)
            raise fnf_error
        except NotADirectoryError as na_dir_error:
            logging.error('Not a directory: %s', ratings_file)
            raise na_dir_error
        except OSError as os_error:
            logging.error('OS error: %s', ratings_file)
            raise os_error

        self.files = [ratings_file + '/' + f for f in files if f.endswith('.csv')]

        # 避免数据过多，进行文件级的采样
        if sample_rate < 1.0 and sample_rate > 0.0:
            self.files = self.files[:int(sample_rate * len(self.files))]
            logging.info('Sampling data files, sampling rate is %s, sampled file num is %s',
                         sample_rate, len(self.files))

        self.current_ratings_frame_len = 0
        self.current_ratings_frame_idx = 0
        self.data_idx = 0
        self.ratings_frame = None

        self._padding_length: int = padding_length
        self._ignore_last_n: int = ignore_last_n
        self._cache = dict()
        self._shift_id_by: int = shift_id_by
        self._chronological: bool = chronological
        self._sample_ratio: float = sample_ratio

        self.seq_columns = seq_columns
        self.nonseq_columns = nonseq_columns
        self.seq_column_names = [k for k, _ in seq_columns.items()]
        self.nonseq_column_names = [k for k, _ in nonseq_columns.items()]
        self.itemid_column_name = itemid_column_name
        self.ratings_column_name = ratings_column_name
        self.timestamps_column_name = timestamps_column_name
        self.column_names = self.seq_column_names + self.nonseq_column_names + \
                            [ratings_column_name, timestamps_column_name]
        if is_train and rank == 0:
            for col in self.seq_column_names:
                logging.info('using seq feature %s', col)
            for col in self.nonseq_column_names:
                logging.info('using nonseq feature %s', col)
            logging.info('ratings column is %s', self.ratings_column_name)
            logging.info('timestamps column is %s', self.timestamps_column_name)
            logging.info('itemid column is %s', self.itemid_column_name)

        self.inner_delim = inner_delim
        self.cut_off_time = cut_off_time
        self.is_train = is_train

        self.time_desc = time_desc
        self.sep = sep
        self.multi_csv_iterator = None
        self._max_candidate_num: int = num_rerank
        self.use_repadding = use_repadding
        if self.use_repadding and rank == 0:
            logging.info('Using sequence expand by repadding')
        self.file_format = file_format
        self.split_chunks_by_worker_id = split_chunks_by_worker_id

    def init_multi_csv_iterator(self) -> None:
        if not self.multi_csv_iterator:
            worker_info = torch.utils.data.get_worker_info()
            self.multi_csv_iterator = MultiCSVIterator(self.files, self.sep, self.rank, self.world_size,
                                                       self.column_names, self.inner_delim,
                                                       itemid_column_name=self.itemid_column_name,
                                                       file_format=self.file_format,
                                                       split_chunks_by_worker_id=self.split_chunks_by_worker_id)

    def __len__(self):
        return DEFAULT_DATA_LEN

    def __iter__(self):
        # 当前df的数据索引
        it = map(self.load_item, self.multi_csv_iterator)
        return it

    def load_item(self, data) -> Dict[str, torch.Tensor]:
        """
        加载单个数据项。

        :param data: 单个数据项。
        :return: 处理后的数据项，包含 history_lengths, test_position, ratings, loss_weights, labels, timestamps
        """

        def eval_as_list(x: str, ignore_last_n) -> List[int]:
            try:
                x = x.replace(self.inner_delim, ',')
            except AttributeError:
                logging.info('%s may not a string, will be converted to str', x)
                x = str(x).replace(self.inner_delim, ',')
            except TypeError as e:
                logging.error('TypeError occurred: %s', e)
                x = str(x).replace(self.inner_delim, ',')
            y = ast.literal_eval(x)
            y_list = [y] if isinstance(y, int) else list(y)
            if self.time_desc:
                y_list.reverse()
            if ignore_last_n > 0:
                # for training data creation
                y_list = y_list[:-ignore_last_n]
            return y_list

        def eval_int_list(x, ignore_last_n: int, shift_id_by: int, sampling_kept_mask: Optional[List[bool]]) -> Tuple[
            List[int], int]:
            y = eval_as_list(x, ignore_last_n=ignore_last_n)
            if sampling_kept_mask is not None:
                y = [x for x, kept in zip(y, sampling_kept_mask) if kept]
            y_len = len(y)
            y.reverse()
            if shift_id_by > 0:
                y = [x + shift_id_by for x in y]
            return y, y_len

        if self._sample_ratio < 1.0:
            raw_length = len(eval_as_list(data[self.itemid_column_name], self._ignore_last_n))
            sampling_kept_mask = (torch.rand((raw_length,), dtype=torch.float32) < self._sample_ratio).tolist()
        else:
            sampling_kept_mask = None

        # 特征处理
        non_seq_feature = {}
        seq_feature_list = {}
        seq_feature_len = {}
        candidate_feature_list = {}

        for column_name in self.seq_column_names:
            seq_feature, seq_history_len = eval_int_list(
                data[column_name], self._ignore_last_n, shift_id_by=self._shift_id_by,
                sampling_kept_mask=sampling_kept_mask
            )
            seq_feature_list[column_name] = seq_feature
            seq_feature_len[column_name] = seq_history_len
        for column_name in self.nonseq_column_names:
            non_seq_feature[column_name] = eval_as_list(data[column_name], self._ignore_last_n)[0]

        ratings, ratings_len = eval_int_list(
            data[self.ratings_column_name], self._ignore_last_n, 0, sampling_kept_mask=sampling_kept_mask
        )
        timestamps, timestamps_len = eval_int_list(
            data[self.timestamps_column_name], self._ignore_last_n, 0, sampling_kept_mask=sampling_kept_mask
        )

        if timestamps_len != ratings_len:
            raise ValueError("timestamps len %s differs from ratings len %s.", (timestamps_len, ratings_len))

        def _truncate_or_pad_seq(y: List[int], target_len: int, chronological: bool) -> List[int]:
            y_len = len(y)
            if y_len < target_len:
                y = y + [0] * (target_len - y_len)
            elif not chronological:
                y = y[:target_len]
            else:
                y = y[-target_len:]
            if len(y) != target_len:
                raise ValueError
            return y

        split_pos = max([i if ts > self.cut_off_time else -1 for i, ts in enumerate(timestamps)]) + 1
        # no test set for this user
        if split_pos == -1:
            split_pos = 0

        candidate_ratings, candidate_timestamps = [], []
        if not self.is_train:
            candidate_timestamps = timestamps[:split_pos]
            candidate_ratings = ratings[:split_pos]

        ratings = ratings[split_pos:]
        timestamps = timestamps[split_pos:]

        if self.use_repadding and self.is_train:
            if len(ratings) == 0 or int(self._padding_length / (len(ratings) + 1)) <= 1:
                pad_num = 0
            else:
                max_pad_num = int(self._padding_length / (len(ratings) + 1))
                pad_num = random.randint(1, max_pad_num)
            orig_ratings = ratings[:]
            orig_timestamps = timestamps[:]
            for _ in range(pad_num):
                ratings = orig_ratings + [-1] + ratings
                timestamps = orig_timestamps + [0] + timestamps

        if self._chronological:
            ratings.reverse()
            timestamps.reverse()
            candidate_timestamps.reverse()
            candidate_ratings.reverse()

        max_seq_len = self._padding_length
        max_candidate_num = self._max_candidate_num

        # aid为64位16进制数字，为了能让torch.int64不溢出，取其前15位
        uid = int(data['column_id'][0:15], 16)
        history_length = min(len(timestamps), max_seq_len)
        ratings = _truncate_or_pad_seq(ratings, max_seq_len, self._chronological)
        timestamps = _truncate_or_pad_seq(timestamps, max_seq_len, self._chronological)

        actual_candidate_num = min(len(candidate_timestamps), max_candidate_num)

        def candidate_feature_interleave(candidate_feature: list):
            return [candidate_feature[i // 2] for i in range(len(candidate_feature) * 2)]

        if not self.is_train:
            candidate_ratings = _truncate_or_pad_seq(candidate_ratings, max_candidate_num, self._chronological)
            candidate_timestamps = _truncate_or_pad_seq(candidate_timestamps, max_candidate_num, self._chronological)

            # this is for evaluation
            # RAB calculation during inference: 
            # truncate bias matrix ---> rel_ts_bias is rel_ts_bias[:, :-num_rerank // 2,:-num_rerank // 2] 
            # during evaluation candidate item and user seq is arranged in this manner 
            # to allow different timestamps for different candidate item to be calculated simultaneously:
            # user_item_1 user_action_1 user_item_2 user_action_2 .... 
            # candidate_item_1 candidate_item_1 candidate_item_2 candidate_item_2 
            candidate_ratings = candidate_feature_interleave(candidate_ratings)
            candidate_timestamps = candidate_timestamps + candidate_timestamps

        for k, v in seq_feature_list.items():
            candidate_v = []
            if not self.is_train:
                candidate_v, v = v[:split_pos], v[split_pos:]
            else:
                v = v[split_pos:]
                if self.use_repadding:
                    orig_v = v[:]
                    for _ in range(pad_num):
                        v = orig_v + [0] + v
            if self._chronological:
                v.reverse()
                candidate_v.reverse()
            v = _truncate_or_pad_seq(v, max_seq_len, self._chronological)
            seq_feature_list[k] = v

            if not self.is_train:
                candidate_v = _truncate_or_pad_seq(candidate_v, max_candidate_num, self._chronological)
                candidate_feature_list['candidate_' + k] = candidate_feature_interleave(candidate_v)

        if self.is_train:
            test_pos = max_seq_len + 1
        else:
            test_pos = min([i if ts > self.cut_off_time else max_seq_len + 1 for i, ts in enumerate(timestamps)])

        # 处理训练数据标签
        labels = ratings

        # 处理模型权重
        loss_weights = [1 for _ in ratings]

        # 间隔处loss weights为0，但是rating需要在embedding内
        if self.use_repadding:
            ratings = [0 if x == -1 else x for x in ratings]

        ret = {
            "uid": torch.tensor(uid, dtype=torch.int64),
            "history_lengths": history_length,
            "test_position": test_pos,
            'ratings': torch.tensor(ratings, dtype=torch.int64),
            "loss_weights": torch.tensor(loss_weights, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            'timestamps': torch.tensor(timestamps, dtype=torch.int64)
        }

        for k, v in non_seq_feature.items():
            ret[k] = torch.tensor(v, dtype=torch.int64)

        for k, v in seq_feature_list.items():
            ret[k] = torch.tensor(v, dtype=torch.int64)

        if not self.is_train:
            candidate_labels = candidate_ratings
            for k, v in candidate_feature_list.items():
                ret[k] = torch.tensor(v, dtype=torch.int64)
            ret["candidate_item_id"] = ret.get("candidate_" + self.itemid_column_name)
            ret["actual_num_rerank"] = actual_candidate_num
            ret["candidate_labels"] = torch.tensor(candidate_labels, dtype=torch.int64)
            ret["candidate_timestamps"] = torch.tensor(candidate_timestamps, dtype=torch.int64)

        return ret


class DatasetV8(IterableDataset):
    """用于处理按时间逆序排列的数据集。"""

    def __init__(
            self,
            ratings_file: str,
            padding_length: int,
            ignore_last_n: int,
            shift_id_by: int = 0,
            chronological: bool = True,
            sample_ratio: float = 1.0,
            is_ads: bool = False,
            sep: str = ',',
            rank: int = 0,
            world_size: int = 1,
            seq_columns=None,
            nonseq_columns=None,
            context_columns=None,
            multi_val_columns=None,
            itemid_column_name='item_id',
            ratings_column_name='event_id',
            timestamps_column_name='time_stamp',
            inner_delim: str = '^',
            is_train: bool = True,
            cut_off_time: int = 1727625600,
            num_rerank=200,
            use_repadding: bool = False,
            token_per_item: int = FeatConst.DFLT_N_TOKEN_PER_ITEM
    ) -> None:
        super().__init__()
        """
        初始化DatasetV8。

        :param ratings_file: 文件路径。
        :param padding_length: 填充长度。
        :param ignore_last_n: 忽略最后n个数据项。
        :param shift_id_by: ID偏移量。
        :param chronological: 默认为True, 处理后按时间正序排列。
        :param sample_ratio: 采样比例。
        :param is_ads: 是否为广告数据。
        :param sep: 分隔符。
        :param rank: 用于分布式训练的rank。
        :param world_size: 用于分布式训练的世界大小。
        :param seq_columns: 序列列名，这里指货/场特征名。
        :param nonseq_columns: 非序列列名，这里指人特征名。
        :param itemid_column_name: 物品ID列名。
        :param ratings_column_name: 评分列名。
        :param timestamps_column_name: 时间戳列名。
        :param inner_delim: 内部分隔符。
        :param is_train: 是否为训练数据。
        :param cut_off_time: 训练/验证切分时间戳, 默认为093000对应的时间戳。
        """

        if seq_columns is None:
            raise ValueError('seq_columns should not be None')
        if nonseq_columns is None:
            raise ValueError('nonseq_columns should not be None')

        self.rank = rank
        self.world_size = world_size

        try:
            files = os.listdir(ratings_file)
        except FileNotFoundError as fnf_error:
            logging.error('No such file or directory: %s', ratings_file)
            raise fnf_error
        except NotADirectoryError as na_dir_error:
            logging.error('Not a directory: %s', ratings_file)
            raise na_dir_error
        except OSError as os_error:
            logging.error('OS error: %s', ratings_file)
            raise os_error

        self.files = [ratings_file + '/' + f for f in files if f.endswith('.csv')]

        self.current_ratings_frame_len = 0
        self.current_ratings_frame_idx = 0
        self.data_idx = 0
        self.ratings_frame = None

        self._padding_length: int = padding_length
        self._ignore_last_n: int = ignore_last_n
        self._cache = dict()
        self._shift_id_by: int = shift_id_by
        self._chronological: bool = chronological
        self._sample_ratio: float = sample_ratio

        self.seq_columns = seq_columns
        self.nonseq_columns = nonseq_columns
        self.context_columns = context_columns
        self.seq_column_names = [k for k, _ in seq_columns.items()]
        self.nonseq_column_names = [k for k, _ in nonseq_columns.items()]
        self.context_column_names = [k for k, _ in context_columns.items()] if context_columns else None
        self.multi_val_columns = multi_val_columns
        self.itemid_column_name = itemid_column_name
        self.ratings_column_name = ratings_column_name
        self.timestamps_column_name = timestamps_column_name
        self.column_names = self.seq_column_names + self.nonseq_column_names + \
                            [ratings_column_name, timestamps_column_name]

        self.inner_delim = inner_delim
        self.cut_off_time = cut_off_time
        self.is_train = is_train

        self.is_ads = is_ads
        self.sep = sep
        self.multi_csv_iterator = None
        self._max_candidate_num: int = num_rerank

        self.use_repadding = use_repadding
        if self.use_repadding and rank == 0:
            logging.info('Using sequence expand by repadding')

        self.token_per_item = token_per_item

    def init_multi_csv_iterator(self) -> None:
        if not self.multi_csv_iterator:
            worker_info = torch.utils.data.get_worker_info()
            self.multi_csv_iterator = MultiCSVIterator(self.files, self.sep, self.rank, self.world_size,
                                                       self.column_names,
                                                       self.inner_delim, itemid_column_name=self.itemid_column_name,
                                                       split_chunks_by_worker_id=False)

    def __len__(self):
        return DEFAULT_DATA_LEN

    def __iter__(self):
        # 当前df的数据索引
        it = map(self.load_item, self.multi_csv_iterator)
        return it

    def _truncate_or_pad_seq(
            self,
            y: List[Union[int, List[int]]],
            target_len: int,
            chronological: bool,
            max_len_per_item: int = 0
    ) -> List[Union[int, List[int]]]:
        """
        将序列 y 调整到长度为 target_len。如果 len_per_item > 0，则假定 y 中的每个元素本身是长度可变的列表，
        先对齐（truncate 或 pad）每个内层列表到长度 len_per_item，再对齐外层列表到长度 target_len；否则仅对齐外层列表。

        参数:
        - y: 原始序列，要么是 List[int]（len_per_item=0），要么是 List[List[int]]（len_per_item>0）。
        - target_len: 目标序列长度。外层长度会被截断或补齐到此值。
        - chronological: 当外层长度大于 target_len 时，决定保留头部还是尾部。
                        True 表示保留最靠近“当前”的末尾 target_len 个元素，False 表示保留最前面的 target_len 个元素。
        - len_per_item: 如果 >0，表示要先把 y 中每个元素（当作列表）对齐到长度 len_per_item，再再做外层对齐。
                        如果 =0，则认为 y 中元素是单个整数，直接对齐外层长度。

        返回值:
        - List[Union[int, List[int]]]: 长度恰为 target_len 的序列；当 len_per_item>0 时，内层列表也固定为长度 len_per_item，
        padding 用 0 填充，截断时取头部（默认）。
        """
        # —— 1. 如果需要对齐内层列表，先对齐每个内层
        if max_len_per_item > 0:
            aligned_inner = []
            for elem in y:
                # 如果原始 elem 不是 list，也把它当作长度 1 的列表处理
                if not isinstance(elem, list):
                    current = [elem]
                else:
                    current = elem.copy()

                # 对齐到 len_per_item: 先截断或填 0
                if len(current) < max_len_per_item:
                    # padding
                    current = current + [0] * (max_len_per_item - len(current))
                elif len(current) > max_len_per_item:
                    # 截断：只保留最前面的 len_per_item 个元素
                    current = current[:max_len_per_item]

                aligned_inner.append(current)
            y = aligned_inner

        # —— 2. 外层长度对齐（truncate or pad）
        y_len = len(y)
        if y_len < target_len:
            if max_len_per_item > 0:
                pad_unit = [0] * max_len_per_item
                pads = [pad_unit for _ in range(target_len - y_len)]
            else:
                pads = [0] * (target_len - y_len)
            y = y + pads

        elif y_len > target_len:
            if chronological:
                # 保留末尾 target_len 个元素
                y = y[-target_len:]
            else:
                # 保留最前面 target_len 个元素
                y = y[:target_len]

        if len(y) != target_len:
            raise ValueError(f"对齐后序列长度不等于 target_len（{target_len}），got {len(y)}")

        return y

    def load_item(self, data) -> Dict[str, torch.Tensor]:
        """
        加载单个数据项。

        :param data: 单个数据项。
        :return: 处理后的数据项，包含 history_lengths, test_position, ratings, loss_weights, labels, timestamps
        """

        def eval_as_list(x: str, ignore_last_n) -> List[int]:

            try:
                if not isinstance(x, str):
                    x = str(x)

                x = x.replace(self.inner_delim, ',')
                #                 print(f"Attempting to parse: {x}")
                # 方法1：使用封装的ast方法
                y = ast.literal_eval(x)
                #                 print("y: ", y)
                #                 # 方法2：自行分割并处理
                #                 y = x.split(",")
                #                 y = [int(i) for i in y]

                if isinstance(y, float):
                    y = int(y)

                y_list = [y] if isinstance(y, int) else list(y)
                if self.is_ads:
                    y_list.reverse()
                if ignore_last_n > 0:
                    # for training data creation
                    y_list = y_list[:-ignore_last_n]

                return y_list

            except Exception as e:
                print("x.type: ", type(x))
                print("x: ", x)
                traceback.print_exc()
                return None

        def eval_int_list(x, ignore_last_n: int, shift_id_by: int, sampling_kept_mask: Optional[List[bool]]) -> Tuple[
            List[int], int]:
            y = eval_as_list(x, ignore_last_n=ignore_last_n)
            if sampling_kept_mask is not None:
                y = [x for x, kept in zip(y, sampling_kept_mask) if kept]
            y_len = len(y)
            y.reverse()
            if shift_id_by > 0:
                y = [x + shift_id_by for x in y]
            return y, y_len

        #         if self._sample_ratio < 1.0:
        #             raw_length = len(eval_as_list(data[self.itemid_column_name], self._ignore_last_n))
        #             sampling_kept_mask = (torch.rand((raw_length,), dtype=torch.float32) < self._sample_ratio).tolist()
        #         else:
        #             sampling_kept_mask = None

        # 针对负样本进行采样
        self._sample_ratio = 1.0
        if self._sample_ratio < 1.0:
            labels = eval_as_list(data[self.ratings_column_name], self._ignore_last_n)
            supervise_list = eval_as_list(data["supervise_list"], self._ignore_last_n)
            sampling_kept_mask = negative_sampling_mask(supervise_list, labels, rate=self._sample_ratio)
        else:
            sampling_kept_mask = None

        # 特征处理
        non_seq_feature = {}
        seq_feature_list = {}
        seq_feature_len = {}
        candidate_feature_list = {}
        len_per_multival = {}

        for column_name in self.seq_column_names:
            seq_feature, seq_history_len = eval_int_list(
                data[column_name], self._ignore_last_n, shift_id_by=self._shift_id_by,
                sampling_kept_mask=sampling_kept_mask
            )
            seq_feature_list[column_name] = seq_feature
            seq_feature_len[column_name] = seq_history_len
            len_per_multival[column_name] = 0

        # print("data: \n", data.head())
        # print("self._ignore_last_n: ", self._ignore_last_n)
        for column_name in self.nonseq_column_names:
            # print(f"Attempting to parse column_name: {column_name}")
            non_seq_feature[column_name] = max(eval_as_list(data[column_name], self._ignore_last_n))

        if self.context_column_names:
            for column_name in self.context_column_names:
                seq_feature, seq_history_len = eval_int_list(
                    data[column_name], self._ignore_last_n, shift_id_by=self._shift_id_by,
                    sampling_kept_mask=sampling_kept_mask
                )
                seq_feature_list[column_name] = seq_feature
                seq_feature_len[column_name] = seq_history_len
                len_per_multival[column_name] = 0

        if self.multi_val_columns:
            for column_name, v in self.multi_val_columns.items():
                seq_feature, seq_history_len = eval_int_list(
                    data[column_name], self._ignore_last_n, shift_id_by=self._shift_id_by,
                    sampling_kept_mask=sampling_kept_mask
                )
                seq_feature_list[column_name] = seq_feature
                seq_feature_len[column_name] = seq_history_len
                len_per_multival[column_name] = v.get("max_len", 0)

        ratings, ratings_len = eval_int_list(
            data[self.ratings_column_name], self._ignore_last_n, 0, sampling_kept_mask=sampling_kept_mask
        )
        timestamps, timestamps_len = eval_int_list(
            data[self.timestamps_column_name], self._ignore_last_n, 0, sampling_kept_mask=sampling_kept_mask
        )

        if timestamps_len != ratings_len:
            raise ValueError("timestamps len %s differs from ratings len %s.", (timestamps_len, ratings_len))

        split_pos = max([i if ts > self.cut_off_time else -1 for i, ts in enumerate(timestamps)]) + 1

        # no test set for this user
        if split_pos == -1:
            split_pos = 0

        candidate_ratings, candidate_timestamps = [], []

        if not self.is_train:
            candidate_timestamps = timestamps[:split_pos]
            candidate_ratings = ratings[:split_pos]

        ratings = ratings[split_pos:]
        timestamps = timestamps[split_pos:]

        if self._chronological:
            ratings.reverse()
            timestamps.reverse()
            candidate_timestamps.reverse()
            candidate_ratings.reverse()

        max_seq_len = self._padding_length
        max_candidate_num = self._max_candidate_num

        history_length = min(len(timestamps), max_seq_len)
        ratings = self._truncate_or_pad_seq(ratings, max_seq_len, self._chronological)
        timestamps = self._truncate_or_pad_seq(timestamps, max_seq_len, self._chronological)

        actual_candidate_num = min(len(candidate_timestamps), max_candidate_num)

        #         print("actual_candidate_num: ", actual_candidate_num)

        def candidate_feature_interleave(candidate_feature: list, n: int = 2) -> list:
            length = len(candidate_feature)
            return [candidate_feature[i // n] for i in range(length * n)]

        if not self.is_train:
            candidate_ratings = self._truncate_or_pad_seq(candidate_ratings, max_candidate_num, self._chronological)
            candidate_timestamps = self._truncate_or_pad_seq(candidate_timestamps, max_candidate_num,
                                                             self._chronological)

            candidate_ratings = candidate_feature_interleave(candidate_ratings, self.token_per_item)

            if self.token_per_item == 2:
                candidate_timestamps = candidate_timestamps + candidate_timestamps

        for k, v in seq_feature_list.items():
            candidate_v = []
            if not self.is_train:
                candidate_v = v[:split_pos]
            v = v[split_pos:]
            if self._chronological:
                v.reverse()
                candidate_v.reverse()
            v = self._truncate_or_pad_seq(v, max_seq_len, self._chronological, len_per_multival[k])
            seq_feature_list[k] = v

            if not self.is_train:
                candidate_v = self._truncate_or_pad_seq(candidate_v, max_candidate_num, self._chronological,
                                                        len_per_multival[k])
                candidate_feature_list['candidate_' + k] = candidate_feature_interleave(candidate_v,
                                                                                        self.token_per_item)

        if self.is_train:
            test_pos = max_seq_len + 1
        else:
            test_pos = min([i if ts > self.cut_off_time else max_seq_len + 1 for i, ts in enumerate(timestamps)])

        #         # 旧方法：处理训练数据标签
        #         label_dict = Const.LABEL_DICT
        #         labels = [label_dict.get(r, 0) for r in ratings]
        #         # 旧方法：处理模型权重
        #         loss_weights = [1 if r in label_dict else 0 for r in ratings]

        # 处理训练数据标签
        labels = ratings

        # 处理模型权重
        loss_weights = [1 for _ in ratings]

        # 间隔处loss weights为0，但是rating需要在embedding内
        if self.use_repadding:
            ratings = [0 if x == -1 else x for x in ratings]

        # aid为64位16进制数字，为了能让torch.int64不溢出，取其前15位
        try:
            # 确保 data['user_id'] 被正确读取,将 user_id 转换为字符串（如果它不是字符串）
            user_id = str(data['user_id'])
            # 取前15个字符并转换为整数
            uid = int(user_id[0:min(15, len(user_id))], 16)

        except Exception as e:
            # 如果发生错误，打印调试信息并设置默认值或抛出异常
            # print(f"Error processing user_id error: {e}")
            uid = 0  # 或者根据需求设置其他默认值

        ret = {
            "uid": torch.tensor(uid, dtype=torch.int64),
            "history_lengths": history_length,
            "test_position": test_pos,
            "ratings": torch.tensor(ratings, dtype=torch.int64),
            "loss_weights": torch.tensor(loss_weights, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "timestamps": torch.tensor(timestamps, dtype=torch.int64),
        }

        for k, v in non_seq_feature.items():
            ret[k] = torch.tensor(v, dtype=torch.int64)

        for k, v in seq_feature_list.items():
            ret[k] = torch.tensor(v, dtype=torch.int64)

        if not self.is_train:
            candidate_labels = candidate_ratings
            for k, v in candidate_feature_list.items():
                ret[k] = torch.tensor(v, dtype=torch.int64)
            ret["candidate_item_id"] = ret.get("candidate_" + self.itemid_column_name)
            ret["actual_num_rerank"] = actual_candidate_num
            ret["candidate_labels"] = torch.tensor(candidate_labels, dtype=torch.int64)
            ret["candidate_timestamps"] = torch.tensor(candidate_timestamps, dtype=torch.int64)

        return ret


class DatasetAG(IterableDataset):
    """用于处理按时间逆序排列的数据集。"""

    def __init__(
            self,
            ratings_file: str,
            ignore_last_n: int,
            shift_id_by: int = 0,
            chronological: bool = True,
            sample_ratio: float = 1.0,
            is_ads: bool = False,
            sep: str = ',',
            rank: int = 0,
            world_size: int = 1,
            history_items_key = None,
            candidate_items_key = None,
            history_feature_columns=None,
            candidate_feature_columns=None,
            nonseq_columns=None,
            itemid_column_name='item_id',
            history_ratings_column_name='history_action_type',
            candidate_ratings_column_name='candidate_action_type',
            history_timestamps_column_name="history_timestamps",
            candidate_timestamps_column_name='candidate_timestamps',
            history_date_column_name='history_date',
            candidate_date_column_name='candidate_date',
            inner_delim: str = '^',
            is_train: bool = True,
            cut_off_time: int = None,
            cut_off_time_lower: int = None,    # ← finetune 新增:train 切分下界(含),YYYYMMDD
            history_length=400,
            num_rerank=400,
            use_repadding: bool = False,
            token_per_item: int = 2,
            use_jagged_data: bool = False
    ) -> None:
        super().__init__()
        """
        初始化DatasetAG。

        :param ratings_file: 文件路径。
        :param ignore_last_n: 忽略最后n个数据项。
        :param shift_id_by: ID偏移量。
        :param chronological: 默认为True, 处理后按时间正序排列。
        :param sample_ratio: 采样比例。
        :param is_ads: 是否为广告数据。
        :param sep: 分隔符。
        :param rank: 用于分布式训练的rank。
        :param world_size: 用于分布式训练的世界大小。
        :param history_feature_columns:
        :param candidate_feature_columns: 注意candidate使用的feature肯能和history不一样
        :param nonseq_columns: 非序列列名，这里指人特征名。
        :param itemid_column_name: 物品ID列名。
        :param ratings_column_name: 评分列名。
        :param timestamps_column_name: 时间戳列名。
        :param inner_delim: 内部分隔符。
        :param is_train: 是否为训练数据。
        :param cut_off_time: 训练/验证切分时间戳, 默认为093000对应的时间戳。
        :param cut_off_time_lower: train 切分下界(含,YYYYMMDD)。仅 finetune 时由 main.py 注入。
                                  非 None 时,train_mask = [cut_off_time_lower <= date < cut_off_time]。
                                  None 时,退化为原行为 train_mask = [date < cut_off_time]。
        """

        if history_feature_columns is None:
            raise ValueError('history_feature_columns should not be None')
        if candidate_feature_columns is None:
            raise ValueError('candidate_feature_columns should not be None')
        if nonseq_columns is None:
            raise ValueError('nonseq_columns should not be None')

        self.rank = rank
        self.world_size = world_size

        try:
            files = os.listdir(ratings_file)
        except FileNotFoundError as fnf_error:
            logging.error('No such file or directory: %s', ratings_file)
            raise fnf_error
        except NotADirectoryError as na_dir_error:
            logging.error('Not a directory: %s', ratings_file)
            raise na_dir_error
        except OSError as os_error:
            logging.error('OS error: %s', ratings_file)
            raise os_error

        self.files = [ratings_file + '/' + f for f in files
                      if f.endswith('.csv') or f.endswith('.csv.gz') or f.endswith('.parquet')]
        
        # get file_format
        if len(self.files) > 0:
            ext_name = None 
            for x in ['.csv', '.csv.gz', '.parquet']:
                if self.files[0].endswith(x):
                    ext_name = x 
            if (ext_name is None) or (not all([f.endswith(ext_name) for f in self.files])):
                raise ValueError('All files in dataset should have a same extention name.')
            
            if ext_name in ['.csv', '.csv.gz']:
                self.file_format = 'csv'
            elif ext_name == '.parquet':
                self.file_format = 'parquet'
            else:
                raise ValueError('Wrong extention name.')
        else:
            self.file_format = 'csv'

        self.current_ratings_frame_len = 0
        self.current_ratings_frame_idx = 0
        self.data_idx = 0
        self.ratings_frame = None

        self._max_history_length = history_length
        self._ignore_last_n: int = ignore_last_n
        self._cache = dict()
        self._shift_id_by: int = shift_id_by
        self._chronological: bool = chronological
        self._sample_ratio: float = sample_ratio

        self.history_items_key = history_items_key
        self.candidate_items_key = candidate_items_key
        self.history_feature_columns = history_feature_columns
        self.candidate_feature_columns = candidate_feature_columns
        self.nonseq_columns = nonseq_columns
        self.history_feature_column_names = [k for k, _ in history_feature_columns.items()]
        self.candidate_feature_column_names = [k for k, _ in candidate_feature_columns.items()]
        self.nonseq_column_names = [k for k, _ in nonseq_columns.items()]
        self.itemid_column_name = itemid_column_name
        self.history_ratings_column_name = history_ratings_column_name
        self.candidate_ratings_column_name = candidate_ratings_column_name
        self.history_timestamps_column_name = history_timestamps_column_name
        self.candidate_timestamps_column_name = candidate_timestamps_column_name
        self.history_date_column_name = history_date_column_name
        self.candidate_date_column_name = candidate_date_column_name

        extra_column_names = [
            self.candidate_ratings_column_name,
            self.history_timestamps_column_name,
            self.candidate_timestamps_column_name,
            self.history_date_column_name,
            self.candidate_date_column_name
        ]
        self.column_names = (
                self.history_feature_column_names +
                self.candidate_feature_column_names +
                self.nonseq_column_names +
                extra_column_names
        )

        self.inner_delim = inner_delim
        self.cut_off_time = cut_off_time
        self.cut_off_time_lower = cut_off_time_lower    # ← finetune 新增
        self.is_train = is_train

        self.is_ads = is_ads
        self.sep = sep
        self.multi_csv_iterator = None
        self._max_candidate_num: int = num_rerank

        self.use_repadding = use_repadding
        if self.use_repadding and rank == 0:
            logging.info('Using sequence expand by repadding')

        self.token_per_item = token_per_item

        self.use_jagged_data = use_jagged_data

    def init_multi_csv_iterator(self) -> None:
        if not self.multi_csv_iterator:
            worker_info = torch.utils.data.get_worker_info()
            self.multi_csv_iterator = MultiCSVIterator(self.files, self.sep, self.rank, self.world_size,
                                                       self.column_names,
                                                       self.inner_delim, itemid_column_name=self.itemid_column_name,
                                                       file_format=self.file_format, 
                                                       split_chunks_by_worker_id=False)

    def __len__(self):
        return DEFAULT_DATA_LEN

    def __iter__(self):
        it = map(self.load_jagged_item, self.multi_csv_iterator)
        return it

    def _truncate_or_pad_seq(
            self,
            y: List[Union[int, List[int]]],
            target_len: int,
            chronological: bool,
            max_len_per_item: int = 1
    ) -> List[Union[int, List[int]]]:
        """
        将序列 y 调整到长度为 target_len。如果 len_per_item > 0，则假定 y 中的每个元素本身是长度可变的列表，
        先对齐（truncate 或 pad）每个内层列表到长度 len_per_item，再对齐外层列表到长度 target_len；否则仅对齐外层列表。

        参数:
        - y: 原始序列，要么是 List[int]（len_per_item=0），要么是 List[List[int]]（len_per_item>0）。
        - target_len: 目标序列长度。外层长度会被截断或补齐到此值。
        - chronological: 当外层长度大于 target_len 时，决定保留头部还是尾部。
                        True 表示保留最靠近"当前"的末尾 target_len 个元素，False 表示保留最前面的 target_len 个元素。
        - len_per_item: 如果 >0，表示要先把 y 中每个元素（当作列表）对齐到长度 len_per_item，再再做外层对齐。
                        如果 =0，则认为 y 中元素是单个整数，直接对齐外层长度。

        返回值:
        - List[Union[int, List[int]]]: 长度恰为 target_len 的序列；当 len_per_item>0 时，内层列表也固定为长度 len_per_item，
        padding 用 0 填充，截断时取头部（默认）。
        """
        # —— 1. 如果需要对齐内层列表，先对齐每个内层
        if max_len_per_item > 1:
            aligned_inner = []
            for elem in y:
                # 如果原始 elem 不是 list，也把它当作长度 1 的列表处理
                if not isinstance(elem, list):
                    current = [elem]
                else:
                    current = elem.copy()

                # 对齐到 len_per_item: 先截断或填 0
                if len(current) < max_len_per_item:
                    # padding
                    current = current + [0] * (max_len_per_item - len(current))
                elif len(current) > max_len_per_item:
                    # 截断：只保留最前面的 len_per_item 个元素
                    current = current[:max_len_per_item]

                aligned_inner.append(current)
            y = aligned_inner

        # —— 2. 外层长度对齐（truncate or pad）
        y_len = len(y)
        if y_len < target_len:
            if max_len_per_item > 1:
                pad_unit = [0] * max_len_per_item
                pads = [pad_unit for _ in range(target_len - y_len)]
            else:
                pads = [0] * (target_len - y_len)
            y = y + pads

        elif y_len > target_len:
            if chronological:
                # 保留末尾 target_len 个元素
                y = y[-target_len:]
            else:
                # 保留最前面 target_len 个元素
                y = y[:target_len]

        if len(y) != target_len:
            raise ValueError(f"对齐后序列长度不等于 target_len（{target_len}），got {len(y)}")

        return y

    def _truncate_or_pad_jagged_seq(
            self,
            y: List[Union[int, List[int]]],
            target_len: int,
            chronological: bool,
            max_len_per_item: int = 1
    ) -> List[Union[int, List[int]]]:
        """
        将序列 y 调整到长度为 target_len。如果 len_per_item > 0，则假定 y 中的每个元素本身是长度可变的列表，
        先对齐（truncate 或 pad）每个内层列表到长度 len_per_item，再对齐外层列表到长度 target_len；否则仅对齐外层列表。

        参数:
        - y: 原始序列，要么是 List[int]（len_per_item=0），要么是 List[List[int]]（len_per_item>0）。
        - target_len: 目标序列长度。外层长度会被截断或补齐到此值。
        - chronological: 当外层长度大于 target_len 时，决定保留头部还是尾部。
                        True 表示保留最靠近"当前"的末尾 target_len 个元素，False 表示保留最前面的 target_len 个元素。
        - len_per_item: 如果 >0，表示要先把 y 中每个元素（当作列表）对齐到长度 len_per_item，再再做外层对齐。
                        如果 =0，则认为 y 中元素是单个整数，直接对齐外层长度。

        返回值:
        - List[Union[int, List[int]]]: 长度恰为 target_len 的序列；当 len_per_item>0 时，内层列表也固定为长度 len_per_item，
        padding 用 0 填充，截断时取头部（默认）。
        """
        # 1.先处理外层截断
        y_len = len(y)
        if y_len > target_len:
            if chronological:
                # 保留末尾 target_len 个元素
                y = y[-target_len:]
            else:
                # 保留最前面 target_len 个元素
                y = y[:target_len]

        # 2. 处理内层，记录内层长度
        multi_per_length =  []
        if max_len_per_item > 1:
            aligned_inner = []
            for elem in y:
                # 如果原始 elem 不是 list，也把它当作长度 1 的列表处理
                if not isinstance(elem, list):
                    current = [elem]
                else:
                    current = elem.copy()
                if len(current) > max_len_per_item:
                    # 截断：只保留最前面的 len_per_item 个元素
                    current = current[:max_len_per_item]
                multi_per_length.append(len(current))
                aligned_inner.append(current)
            y = aligned_inner
        return y, multi_per_length


    def load_jagged_item(self, data) -> Dict[str, torch.Tensor]:
        """
        加载单个数据项。

        :param data: 单个数据项。
        :return: 处理后的数据项，
        """

        def eval_as_list(x: str, ignore_last_n) -> List[int]:

            try:
                if not isinstance(x, str):
                    x = str(x)

                x = x.replace(self.inner_delim, ',')
                y = ast.literal_eval(x)

                y_list = [y] if isinstance(y, int) else list(y)
                if not self._chronological:  # 全部变成时序升序排列
                    y_list.reverse()
                if ignore_last_n > 0:
                    # for training data creation，去掉最后n个最近的数据
                    y_list = y_list[:-ignore_last_n]

                return y_list

            except Exception as e:
                print("x.type: ", type(x))
                print("x: ", x)
                traceback.print_exc()
                return None
        
        def eval_list(x, ignore_last_n: int, 
                      shift_id_by: int, sampling_kept_mask: Optional[List[bool]] = None) -> Tuple[List[int], int]:
            y = eval_as_list(x, ignore_last_n=ignore_last_n)
            if sampling_kept_mask is not None:
                y = [x for x, kept in zip(y, sampling_kept_mask) if kept]
            y_len = len(y)
            if shift_id_by > 0:
                y = [x + shift_id_by for x in y]
            return y, y_len

        try: 
            ###############################################
            # 读取特征
            # 读取user 特征
            non_seq_feature = {}
            for column_name in self.nonseq_column_names:
                # print(f"Attempting to parse column_name: {column_name}")
                non_seq_feature[column_name] = max(eval_as_list(data[column_name], self._ignore_last_n))

            max_len_per_multival = {}  # 每个特征的最大长度，通过config读取，单值特征为1，多值特征为n
            # 读取history部分
            history_feature_list = {}
            history_feature_len = {}
            history_feature_multi_len = {}
            history_feature_multi_sum_len = {}
            for column_name in self.history_feature_column_names:
                dtype = self.history_feature_columns[column_name]['dtype']
                seq_feature, seq_len = eval_list(data[column_name], self._ignore_last_n, 
                                                     shift_id_by=(0 if dtype == 'con' else self._shift_id_by))
                history_feature_list[column_name] = seq_feature
                history_feature_len[column_name] = seq_len
                feat_max_len = self.history_feature_columns.get(column_name).get("max_len", 1)
                max_len_per_multival[column_name] = feat_max_len

            history_timestamps, _ = eval_list(
                data[self.history_timestamps_column_name], self._ignore_last_n, 0)
            history_ratings, _ = eval_list(
                data[self.history_ratings_column_name], self._ignore_last_n, 0)
            history_dates, _ = eval_list(
                data[self.history_date_column_name], self._ignore_last_n, 0)

            # 读取candidate部分
            candidate_feature_list = {}
            candidate_feature_len = {}
            candidate_feature_multi_len = {}
            candidate_feature_multi_sum_len = {}
            for column_name in self.candidate_feature_column_names:
                dtype = self.candidate_feature_columns[column_name]['dtype']
                seq_feature, seq_len = eval_list(data[column_name], self._ignore_last_n, 
                                                     shift_id_by=(0 if dtype == 'con' else self._shift_id_by))
                candidate_feature_list[column_name] = seq_feature
                candidate_feature_len[column_name] = seq_len
                feat_max_len = self.candidate_feature_columns.get(column_name).get("max_len", 1)
                max_len_per_multival[column_name] = feat_max_len

            candidate_ratings, candidate_ratings_len = eval_list(
                data[self.candidate_ratings_column_name], self._ignore_last_n, 0)
            candidate_timestamps, candidate_timestamps_len = eval_list(
                data[self.candidate_timestamps_column_name], self._ignore_last_n, 0)
            candidate_dates, _ = eval_list(
                data[self.candidate_date_column_name], self._ignore_last_n, 0)

            if candidate_timestamps_len != candidate_ratings_len:
                raise ValueError(
                    f"timestamps len {candidate_timestamps_len} differs from ratings len {candidate_ratings_len}.")

            #################################################################
            # 对于candidate, 根据date分割train和test, 直接设置对应的掩码（现在要求cut_off_time为date格式，yymmdd）
            # train_mask 支持范围切分:[cut_off_time_lower, cut_off_time)。
            # 当 cut_off_time_lower 为 None 时,退化为原始行为(cand_date < cut_off_time)。
            # 这是 finetune 双 cut_off 方案的核心:
            #   - cut_off_time       = data_dir 路径日期(train 上界,eval 下界)
            #   - cut_off_time_lower = load_data_dir 路径日期 - cut_off_bias(train 下界)
            if self.cut_off_time_lower is not None:
                train_mask = [self.cut_off_time_lower <= cand_date < self.cut_off_time
                              for cand_date in candidate_dates]
            else:
                train_mask = [cand_date < self.cut_off_time for cand_date in candidate_dates]
            test_mask = [cand_date >= self.cut_off_time for cand_date in candidate_dates]
            history_mask = [hist_date < self.cut_off_time for hist_date in history_dates]

            if not self.is_train:  # test阶段
                candidate_timestamps = list(compress(candidate_timestamps, test_mask))
                if len(candidate_timestamps) == 0:
                    candidate_timestamps = [0]                
                # print('len(candidate_timestamps)', len(candidate_timestamps))
                candidate_ratings = list(compress(candidate_ratings, test_mask))
                if len(candidate_ratings) == 0:
                    candidate_ratings = [0] 
                candidate_dates = list(compress(candidate_dates, test_mask))
                if len(candidate_dates) == 0:
                    candidate_dates = [0] 
                for k, v in candidate_feature_list.items():
                    v = list(compress(v, test_mask))
                    candidate_feature_list[k] = [0] if len(v) == 0 else v
            else:  # train阶段
                candidate_timestamps = list(compress(candidate_timestamps, train_mask))
                if len(candidate_timestamps) == 0:
                    candidate_timestamps = [0]                
                candidate_ratings = list(compress(candidate_ratings, train_mask))
                if len(candidate_ratings) == 0:
                    candidate_ratings = [0] 
                candidate_dates = list(compress(candidate_dates, train_mask))
                if len(candidate_dates) == 0:
                    candidate_dates = [0] 
                for k, v in candidate_feature_list.items():
                    v = list(compress(v, train_mask))
                    candidate_feature_list[k] = [0] if len(v) == 0 else v
                               
            # history_timestamps = list(compress(history_timestamps, history_mask))
            # history_ratings = list(compress(history_ratings, history_mask))
            # history_dates = list(compress(history_dates, history_mask))
            # for k, v in history_feature_list.items():
            #     v = list(compress(v, history_mask))
            #     history_feature_list[k] = v

            ####################################################
            # 处理序列长度，padding
            def candidate_feature_interleave(candidate_feature: list, n: int = 2) -> list:
                length = len(candidate_feature)
                return [candidate_feature[i // n] for i in range(length * n)]

            #0610数据，从左到右是从旧到新
            # 处理history部分
            # if len(history_timestamps) == 0:
            #     history_timestamps = [0]
            #     for k, v in history_feature_list.items():
            #         history_feature_list[k] = [0]
            history_length = min(len(history_timestamps), self._max_history_length)
            history_timestamps = self._truncate_or_pad_seq(history_timestamps, self._max_history_length,
                                                        True)
            history_ratings = self._truncate_or_pad_seq(history_ratings, self._max_history_length,
                                                        True)
            history_dates = self._truncate_or_pad_seq(history_dates, self._max_history_length,
                                                    True)

            for k, v in history_feature_list.items():
                v, v_multi_per_length= self._truncate_or_pad_jagged_seq(v, self._max_history_length, True, max_len_per_multival[k])
                if max_len_per_multival[k] > 1: 
                    history_feature_multi_len[k] = v_multi_per_length
                    history_feature_multi_sum_len[k] = sum(v_multi_per_length)
                    history_feature_list[k] = sum(v, [])
                else:
                    history_feature_list[k] = v

            # 处理candidate部分
            # if len(candidate_timestamps) == 0:
            #     candidate_timestamps = [0]
            #     for k, v in candidate_feature_list.items():
            #         candidate_feature_list[k] = [0]
            max_candidate_num = self._max_candidate_num
            actual_candidate_num = min(len(candidate_timestamps), max_candidate_num)
            candidate_ratings = self._truncate_or_pad_seq(candidate_ratings, max_candidate_num, True)
            candidate_ratings = candidate_feature_interleave(candidate_ratings, self.token_per_item)
            candidate_timestamps = self._truncate_or_pad_seq(candidate_timestamps, max_candidate_num, True)
            candidate_dates = self._truncate_or_pad_seq(candidate_dates, max_candidate_num, True)
            if self.token_per_item == 2:
                candidate_timestamps = candidate_timestamps + candidate_timestamps
                candidate_dates = candidate_dates + candidate_dates

            for k, v in candidate_feature_list.items():
                v, v_multi_per_length = self._truncate_or_pad_jagged_seq(v, self._max_candidate_num, True, max_len_per_multival[k])
                v = candidate_feature_interleave(v, self.token_per_item)
                if max_len_per_multival[k] > 1: 
                    candidate_feature_multi_len[k] = v_multi_per_length
                    candidate_feature_multi_sum_len[k] = sum(v_multi_per_length)
                    candidate_feature_list[k] = sum(v, [])
                else:
                    candidate_feature_list[k] = v

            #######################################################
            # 处理训练数据标签
            labels = candidate_ratings  # 新数据方案candidate_action_type本身只有0曝光1下载，直接可以用做ground_truth

            # 处理模型权重
            loss_weights = [1 if t != 0 else 0 for t in candidate_timestamps]
            # print('loss_weights: ', loss_weights)
            
            history_ids = self._truncate_or_pad_seq(history_feature_list[self.history_items_key], 
                                                    self._max_history_length, True)
            candidate_ids = self._truncate_or_pad_seq(candidate_feature_list[self.candidate_items_key], 
                                                      max_candidate_num, True)

            # 间隔处loss weights为0，但是rating需要在embedding内
            if self.use_repadding:  # 间隔处是啥意思？没看懂: 猜测是补零位
                candidate_ratings = [0 if x == -1 else x for x in candidate_ratings]

            # aid为64位16进制数字，为了能让torch.int64不溢出，取其前15位
            try:
                # 确保 data['user_id'] 被正确读取,将 user_id 转换为字符串（如果它不是字符串）
                user_id = str(data['device_id_sha256'])
                # 取前15个字符并转换为整数
                uid = int(user_id[0:min(15, len(user_id))], 16)

            except Exception as e:
                # 如果发生错误，打印调试信息并设置默认值或抛出异常
                # print(f"Error processing user_id error: {e}")
                uid = 0  # 或者根据需求设置其他默认值
            ret = {
                "uid": torch.tensor(uid, dtype=torch.int64),
                "history_lengths": history_length,
                "candidate_lengths": actual_candidate_num,
                "history_action_type": torch.tensor(history_ratings, dtype=torch.int64).unsqueeze(0),
                "candidate_action_type": torch.tensor(candidate_ratings, dtype=torch.int64).unsqueeze(0),
                "loss_weights": torch.tensor(loss_weights, dtype=torch.float32).unsqueeze(0),
                "labels": torch.tensor(labels, dtype=torch.int64).unsqueeze(0),
                "history_ids": torch.tensor(history_ids, dtype=torch.int64).unsqueeze(0),
                "candidate_ids": torch.tensor(candidate_ids, dtype=torch.int64).unsqueeze(0),
                self.history_timestamps_column_name: torch.tensor(history_timestamps, dtype=torch.int64).unsqueeze(0),
                self.candidate_timestamps_column_name: torch.tensor(candidate_timestamps, dtype=torch.int64).unsqueeze(0),
                self.history_date_column_name: torch.tensor(history_dates, dtype=torch.int64).unsqueeze(0),
                self.candidate_date_column_name: torch.tensor(candidate_dates, dtype=torch.int64).unsqueeze(0)
            }

            for k, v in non_seq_feature.items():
                dtype = self.nonseq_columns[k]['dtype']
                ret[k] = torch.tensor(v, dtype=(torch.float if dtype == 'con' else torch.int64))

            for k, v in history_feature_list.items():
                dtype = self.history_feature_columns[k]['dtype']
                ret[k] = torch.tensor(v, dtype=(torch.float if dtype == 'con' else torch.int64))
                if k in history_feature_multi_len.keys():
                    ret[ k + '_multi_len'] = torch.tensor(history_feature_multi_len[k], dtype=torch.int64)
                    ret[ k + '_multi_sum_len'] = torch.tensor(history_feature_multi_sum_len[k], dtype=torch.int64)

            for k, v in candidate_feature_list.items():
                dtype = self.candidate_feature_columns[k]['dtype']
                ret[k] = torch.tensor(v, dtype=(torch.float if dtype == 'con' else torch.int64))
                if k in candidate_feature_multi_len.keys():
                    ret[ k + '_multi_len'] = torch.tensor(candidate_feature_multi_len[k], dtype=torch.int64)
                    ret[ k + '_multi_sum_len'] = torch.tensor(candidate_feature_multi_sum_len[k], dtype=torch.int64)
            return ret
        
        except:
            raise ValueError('Error in load_jagged_item.')

def negative_sampling_mask(data1_supervise, data2_label, rate=0.07):
    """
    根据data1和data2生成掩码，并根据指定的采样率进行负采样。
    1. 目标榜单的训测数据
    2. 首先属于目标榜单（暂略）
    3. 然后supervise_list = 1,
    4. 然后负样本 0.07采样路采样

    :param data1_supervise: 标识是否用于监督训练的数据（0为否，1为是）
    :param data2_label: 用户的行为数据（0为负样本，非0为正样本）
    :param rate: 负采样的比率，默认是0.07。
    :return: 生成的掩码数组。
    """
    data1_supervise = np.array(data1_supervise)
    data2_label = np.array(data2_label)
    assert len(data1_supervise) == len(data2_label), "data1和data2的长度必须相等"
    N = len(data1_supervise)

    # 初始化掩码为全1（True）
    mask = np.ones(N, dtype=bool)

    # 按照要求设置部分mask值为0（False）
    mask[(data1_supervise == 1) & (data2_label == 0)] = False

    # 获取需要进行负采样的索引列表
    false_indices = np.where(mask == False)[0]

    # 计算采样数量
    sample_num = round(len(false_indices) * rate)

    # 确保至少采样一个
    if sample_num >= 1:
        # 随机选择索引进行采样
        sampled_indices = np.random.choice(false_indices, size=sample_num, replace=False)

        #         # 更新掩码：只保留采样出来的索引为False，其余设为True
        #         mask.fill(True)
        #         mask[sampled_indices] = False

        # 更新掩码：将所有 false_indices 设为 False，然后只将 sampled_indices 设为 True
        mask.fill(True)
        mask[false_indices] = False
        mask[sampled_indices] = True

    else:
        # 如果没有符合条件的索引，则保持初始掩码不变或根据需求调整
        mask = None
    #         print("没有找到符合条件的索引进行负采样。")

    return mask
