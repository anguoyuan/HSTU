import os

import torch

from data.concat_dataset import DatasetMusicV1


def create_data_loader(
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        prefetch_factor: int = 128,
        num_workers: int = os.cpu_count(),
) -> torch.utils.data.DataLoader:
    """
    创建一个数据加载器(DataLoader), 用于批量加载数据集。

    :param dataset: 要加载的数据集。
    :param batch_size: 每个批次的样本数量。
    :param prefetch_factor: 预取因子，用于控制预取的数据量。
    :param num_workers: 加载数据时使用的子进程数量, 默认为CPU核心数。
    :return: 一个配置好的DataLoader对象。
    """

    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        ds: DatasetMusicV1 = worker_info.dataset
        ds.files = ds.files[worker_id::worker_info.num_workers]
        ds.init_multi_csv_iterator()

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=worker_init_fn,
        drop_last=True
    )

    return data_loader


def create_data_loader_ep(
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        prefetch_factor: int = 128,
        num_workers: int = os.cpu_count(),
) -> torch.utils.data.DataLoader:
    """
    创建一个数据加载器(DataLoader), 用于批量加载数据集。

    :param dataset: 要加载的数据集。
    :param batch_size: 每个批次的样本数量。
    :param prefetch_factor: 预取因子，用于控制预取的数据量。
    :param num_workers: 加载数据时使用的子进程数量, 默认为CPU核心数。
    :return: 一个配置好的DataLoader对象。
    """

    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        ds: DatasetMusicV1 = worker_info.dataset
        ds.files = ds.files[worker_id::worker_info.num_workers]
        ds.init_multi_csv_iterator()

    def collect_fn(data):
        """
            udis:储存data中所有uid信息
            past_lengths:储存data中所有history_lengths信息
            pass_pagloads:
        """

        past_payloads = {}
        for row in data:
            for key, val in row.items():
                past_payloads.setdefault(key, []).append(val)

        for key, lst in past_payloads.items():
            if len(lst) > 0 and isinstance(lst[0], torch.Tensor) and lst[0].dim() > 0:
                past_payloads[key] = torch.concat(lst) 
            else:
                past_payloads[key] = torch.tensor(lst)
            
        return past_payloads

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collect_fn,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=worker_init_fn,
        drop_last=True
    )
    return data_loader
