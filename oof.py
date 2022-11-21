# -*- coding: utf-8 -*-
# @Time : 2022/4/28 下午5:30
# @Author : YANG.C
# @File : oof.py
import os
import sys
import numpy as np
import faiss
import shutil
import torch
import random


def create_and_search_index(embedding_size: int, train_embeddings: np.ndarray, val_embeddings: np.ndarray, k: int):
    index = faiss.IndexFlatIP(embedding_size)
    index.add(train_embeddings)
    D, I = index.search(val_embeddings, k=k)  # noqa: E741

    return D, I


def main(dirs):
    train_image_names, train_embeddings, train_targets = np.load(f'{dirs}/train_name.npy', allow_pickle=True), \
                                                         np.load(f'{dirs}/train_feature.npy', allow_pickle=True), \
                                                         np.load(f'{dirs}/train_targets.npy', allow_pickle=True)

    val_image_names, val_embeddings, val_targets, val_logits = np.load(f'{dirs}/val_name.npy', allow_pickle=True), \
                                                               np.load(f'{dirs}/val_feature.npy', allow_pickle=True), \
                                                               np.load(f'{dirs}/val_targets.npy', allow_pickle=True), \
                                                               np.load(f'{dirs}/val_logits.npy', allow_pickle=True)

    k = 50
    dims = 512
    D, I = create_and_search_index(dims, train_embeddings, val_embeddings, k)
    cor, total = 0, 0
    logit = 0
    errorSet = {}
    for i in range(0, 17):
        errorSet[i] = []

    for i in range(len(D)):
        # print(train_targets[I[i]])
        # print(val_logits[i][train_targets[I[i]]])
        # D[i] += val_logits[i][train_targets[I[i]]]
        # print(D[i])
        # exit(0)
        # for j in range(D[i]):
        #     pass
        # print(D[i])
        # print(I[i])
        # exit(0)

        D[i] = D[i] + val_logits[i][train_targets[I[i]]]
        similar = np.argmax(D[i])
        classIdx = I[i][similar]
        if val_targets[i] == train_targets[classIdx]:
            cor += 1
        else:
            errorSet[val_targets[i] + 1].append(val_image_names[i])
            # print(f'/opt/datasets/Yaoyao/test_edge/{val_targets[i] + 1}/{val_image_names[i]}')
        total += 1
        # if val_targets[i] == train_targets[I[i][0]]:
        #     # print(i)
        #     cor += 1
        # if val_targets[i] == np.argmax(val_logits[i]):
        #     logit += 1
        # total += 1

    print('acc: ', cor / total)
    srcRoot = '/opt/datasets/Yaoyao'
    srcTrain = f'{srcRoot}/train20'
    srcVal = f'{srcRoot}/test_edge'
    tmp = f'{srcRoot}/test_edge/tmp2'

    for k, v in errorSet.items():
        val2Train = []
        train2Val = []
        if k == 0:
            continue
        vts = os.listdir(f'{srcTrain}/{k}')
        if len(vts) < len(v):
            continue

        for vv in errorSet[k]:
            val2Train.append(f'{srcVal}/{k}/{vv}')

        vts = os.listdir(f'{srcTrain}/{k}')
        vt = random.sample(vts, len(v))
        for vti in vt:
            train2Val.append(f'{srcTrain}/{k}/{vti}')

        val2Train.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
        print(val2Train, len(val2Train))
        if k == 2:
            for vt in val2Train:
                shutil.move(vt, f'{tmp}')
            exit(0)
        # print(train2Val, len(train2Val))
        # for tv in train2Val:
        #     if os.path.exists(tv) and os.path.exists(f'{srcVal}/{k}/{tv.split("/")[-1]}'):
        #         continue
        #     shutil.move(tv, f'{srcVal}/{k}')
        # for vt in val2Train:
        #     if os.path.exists(vt) and os.path.exists(f'{srcTrain}/{k}/{vt.split("/")[-1]}'):
        #         continue
        #     shutil.move(vt, f'{srcTrain}/{k}')
        # print(' + ' * 10)


if __name__ == '__main__':
    dirs = sys.argv[1]
    main(dirs)
