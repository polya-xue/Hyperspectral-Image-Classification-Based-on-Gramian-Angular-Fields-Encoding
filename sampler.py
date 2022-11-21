import copy
import itertools
from collections import defaultdict
from typing import Optional, List
import random

import numpy as np
from torch.utils.data.sampler import Sampler


# class NaiveIdentitySampler(Sampler):
#     """
#     Randomly sample N identities, then for each identity,
#     randomly sample K instances, therefore batch size is N*K.
#     Args:
#     - data_source (list): list of (img_path, pid, camid).
#     - num_instances (int): number of instances per identity in a batch.
#     - batch_size (int): number of examples in a batch.
#     """
#     def __init__(self, ):


class RandomIdentiyiSampler(Sampler):  # 这里是设置了如何将每个mini_batch按照N×K进行选取的策略
    def __init__(self, data_source, batch_size, num_instance):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instance = num_instance
        self.num_pids_per_batch = self.batch_size // self.num_instance  # N*K = Batch_size
        self.index_dic = defaultdict(list)

        # 为每个 class 添加 img_path(index)
        for index, pid in enumerate(self.data_source):
            self.index_dic[pid].append(index)

        # class 个数
        self.pids = list(self.index_dic.keys())

        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instance:
                num = self.num_instance

            # num == num_instance, 本身个数
            # num < num_instance,  num 个数
            # num > num_instance,  num_instance 倍数的个数
            self.length += num - num % self.num_instance

    def __iter__(self):
        # 每个类别选出预设的img_path
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            # 当前类别的所有img_path(idxs)
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instance:
                idxs = np.random.choice(idxs, size=self.num_instance, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instance:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            # 选择到的 pids
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.lenght = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length
