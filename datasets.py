import os.path
import random
from collections import defaultdict

import cv2
from PIL import Image
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import transforms
from torch_utils import torch_distributed_zero_first


def my_collect_fn(batch_list):
    img_list, img_name_list, target_list = [], [], []
    tmp_image_list = []
    tmp_image_name_list = []
    tmp_target_list = []
    for i in range(len(batch_list)):
        for j in range(len(batch_list[i])):
            image_name, image, target = batch_list[i][j]['image_name'], batch_list[i][j]['image'], batch_list[i][j][
                'target']
            tmp_image_list.append(image.unsqueeze(0))
            tmp_image_name_list.append(image_name)
            tmp_target_list.append(target.unsqueeze(0))

    abc = list(zip(tmp_image_list, tmp_image_name_list, tmp_target_list))
    random.shuffle(abc)
    tmp_image_list[:], tmp_image_name_list[:], tmp_target_list[:] = zip(*abc)

    # print(tmp_image_name_list)

    return {'image': torch.cat(tmp_image_list, dim=0),
            'image_name': tmp_image_name_list,
            'target': torch.cat(tmp_target_list, dim=0),
            }


def create_dataloader(pipeline, rank=-1, train=False, val=False, test=False, all_train=False, triplet=False):
    dataloader, dataset, transforms, fold = pipeline['dataloader'], pipeline['dataset'], \
                                            pipeline['transforms'], pipeline['fold']
    batch_size = dataloader['batch_size']
    num_workers = dataloader['num_workers']
    drop_last = dataloader['drop_last']
    pin_memory = dataloader['pin_memory']
    shuffle = dataloader['shuffle']

    root_dir = dataset['root_dir']

    with torch_distributed_zero_first(rank):
        if not triplet:
            datasets = MyDataset(root_dir,
                                 transform=transforms)
        else:
            datasets = MyTripletDataset(root_dir, transform=transforms)

    sampler = torch.utils.data.distributed.DistributedSampler(datasets) if rank != -1 else None
    dataloader = torch.utils.data.DataLoader(datasets,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             sampler=sampler,
                                             pin_memory=pin_memory,
                                             drop_last=drop_last,
                                             shuffle=shuffle,
                                             collate_fn=my_collect_fn if triplet else None
                                             )
    return dataloader, datasets


class HappyWhaleDatasetForRetrieval(Dataset):
    def __init__(self,
                 root_dir,
                 # data_info,
                 transform=None):
        super(HappyWhaleDatasetForRetrieval, self).__init__()
        self.root_dir = root_dir
        self.all_images = os.listdir(self.root_dir)
        # self.data_info = data_info
        self.transform = transform

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        img_file = f'{self.root_dir}/{self.all_images[index]}'
        try:
            img = cv2.imread(img_file)
            img = img[:, :, ::-1].copy()  # BGR2RGB
        except Exception as e:
            print(f'Image cannot be opened (index {index}, file {img_file}). {str(e)}')
            raise e

        if self.transform is not None:
            img = self.transform(img)
        return {
            "image": img,
            'image_name': self.all_images[index],
        }


class MyDataset(Dataset):
    def __init__(self,
                 root_dir,
                 transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        # 设置类别索引
        classes = os.listdir(self.root_dir)
        classes.sort(key=lambda c: int(c))
        self.class2idx = {}
        class_idx = 0
        for c in classes:
            self.class2idx[c] = class_idx
            class_idx += 1

        # 加载数据路径和类别标签
        self.data = []
        self.label = []
        for path in os.listdir(self.root_dir):
            for img_name in os.listdir(f'{self.root_dir}/{path}'):
                self.data.append(f'{self.root_dir}/{path}/{img_name}')
                self.label.append(self.class2idx[path])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, target = self.data[index], self.label[index]

        img = cv2.imread(img_file)
        img = img[:, :, ::-1].copy()  # BGR2RGB

        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = -1
        return {
            'image': img,
            'image_name': img_file.split('/')[-1],
            'target': target,
        }


class MyTripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super(MyTripletDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform

        # 设置类别索引
        classes = os.listdir(root_dir)
        classes.sort(key=lambda c: int(c))
        self.class2idx = {}
        class_idx = 0
        self.pid2path = defaultdict(list)
        for c in classes:
            self.class2idx[c] = class_idx
            self.pid2path[class_idx] = []
            class_idx += 1

        # 加载数据路径和类别标签
        self.data = []
        self.label = []

        for path in os.listdir(self.root_dir):
            for img_name in os.listdir(f'{self.root_dir}/{path}'):
                self.data.append(f'{self.root_dir}/{path}/{img_name}')
                self.label.append(self.class2idx[path])
                self.pid2path[self.class2idx[path]].append(f'{self.root_dir}/{path}/{img_name}')

        self.batch_size = 4
        self.num_instance = 2
        self.num_pids_per_batch = self.batch_size // self.num_instance

        self.unique_targets = list(self.class2idx.values())
        self.num_all_classes = len(self.unique_targets)

        self.step_all = len(self.unique_targets) // self.num_pids_per_batch
        self.read_order = random.sample(self.unique_targets, self.num_all_classes)
        print(f'num_all_classes: {self.num_all_classes}, step_all: {self.step_all}')

    def shuffle(self):
        self.read_order = random.sample(self.unique_targets, self.num_all_classes)

    def getitem(self, img_path, target):
        image_name = img_path.split('/')[-1]
        image_path = img_path
        image = cv2.imread(image_path)
        image = image[:, :, ::-1].copy()

        # image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        target = target
        target = torch.tensor(target, dtype=torch.long)

        return {"image_name": image_name, "image": image, "target": target}

    def __getitem__(self, step):
        if step > self.step_all - 1:
            print('step train out of size')

        class_ids = self.read_order[step * self.num_pids_per_batch: (step + 1) * self.num_pids_per_batch]
        imgs_tmp = []
        for class_id in class_ids:
            num = min(self.num_instance, len(self.pid2path[class_id]))
            # print(f'num: {num}, class_id: {class_id}')
            tmp_img_path = self.pid2path[class_id]
            # print(f'tmp_img_path-start: {tmp_img_path}')

            if num < self.num_instance:
                img_idx = np.random.choice(len(tmp_img_path), self.num_instance - num)
                for img in img_idx:
                    tmp_img_path.append(self.pid2path[class_id][img])
            else:
                img_idx = np.random.choice(len(tmp_img_path), self.num_instance)
                tmp_img_path = []
                for img in img_idx:
                    tmp_img_path.append(self.pid2path[class_id][img])

            # print(f'tmp_img_path-ended: {tmp_img_path}')
            for img_path in tmp_img_path:
                img_tmp = self.getitem(img_path, class_id)
                imgs_tmp.append(img_tmp)
        return imgs_tmp

    def __len__(self) -> int:
        return self.step_all
