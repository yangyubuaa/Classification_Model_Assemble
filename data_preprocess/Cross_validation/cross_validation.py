# -*- utf-8 -*-
# @Time: 2021/4/20 10:15 上午
# @Author: yang yu
# @File: cross_validation.py.py
# @Software: PyCharm
import torch
import copy

class KFoldCrossValidation:
    '''k折交叉验证
    输入为data_process.Dataset.dataset中重写的dataset类
    输出也是相同的dataset类
    '''
    def __init__(self, k_fold=10):
        self.k_fold = k_fold
        self.split_thres = 0.8

    def __call__(self, dataset, shuffle=True):
        '''K Fold 交叉验证

        将数据集等分为k份，分别将每个数据集作为测试集，其他数据整合为训练数据，返回k个数据集的生成器

        :param dataset:
        :return: generator
        '''
        if shuffle:
            dataset.shuffle_()

        if self.k_fold != 0:
            data_nums = len(dataset)
            sp_thres = int(data_nums / self.k_fold)
            index = 0
            # average_seperate中存储的是分割数据集
            average_seperate = list()
            for i in range(0, data_nums, sp_thres):
                if index < self.k_fold:
                    part_dataset = dataset.slice(i, i+sp_thres)
                    average_seperate.append(part_dataset)
                    index = index + 1
                else:
                    break

            average_seperate_sum = len(average_seperate)
            for index, eval_set in enumerate(average_seperate):
                train_select = [i for i in range(average_seperate_sum)]
                train_select.remove(index)
                train_dataset = copy.deepcopy(average_seperate[train_select[0]])
                # print(train_dataset)
                eval_dataset = average_seperate[index]
                for i in train_select[1:]:
                    # print(average_seperate[i])
                    train_dataset.extend_(average_seperate[i])

                yield (train_dataset, eval_dataset)
        else:
            data_nums = len(dataset)
            split_point = int(data_nums * self.split_thres)
            yield (dataset.slice(0, split_point), dataset.slice(split_point, data_nums))


if __name__ == '__main__':
    # 测试
    pass
