# -*- coding: utf-8 -*-
# @Time : 2022/5/11 下午5:08
# @Author : YANG.C
# @File : move.py

import os
import random
import shutil
import sys


def main(src, dst):
    trainRoot = '/media/mclab207/Datas/yc/Yaoyao/train20'
    testRoot = '/media/mclab207/Datas/yc/Yaoyao/test_edge'
    dirs = [4, 9, 15]
    addNums = [16, 10, 30]

    for i in range(3):
        trainImgs = os.listdir(f'{trainRoot}/{str(dirs[i])}')
        testImgs = os.listdir(f'{testRoot}/{str(dirs[i])}')

        print(len(trainImgs), len(testImgs))

        test2train = random.sample(testImgs, addNums[i])

        for imgName in test2train:
            try:
                print(f'{testRoot}/{dirs[i]}/{imgName}', f'{trainRoot}/{dirs[i]}/')
                # shutil.move(f'{testRoot}/{dirs[i]}/{imgName}', f'{trainRoot}/{dirs[i]}/')
            except Exception:
                print(f'{trainRoot}/{dirs[i]}/{imgName} already exists!')


if __name__ == '__main__':
    srcPath = sys.argv[1]
    dstPath = sys.argv[2]
    main(srcPath, dstPath)
