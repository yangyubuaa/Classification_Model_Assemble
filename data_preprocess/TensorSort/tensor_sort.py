# -*- utf-8 -*-
# @Time: 2021/4/21 9:56 上午
# @Author: yang yu
# @File: tensor_sort.py.py
# @Software: PyCharm

import torch

def tensor_seq_len_desent(tensor_in, seq_len_in, label_in):
    '''将输入的tensor按照seq_len从大到小重新排序
    :param tensor_in:
    :param seq_len_in:
    :return:
    '''
    seq_len_out, idx_sort = torch.sort(seq_len_in, dim=0, descending=True)
    idx_sort = idx_sort.squeeze()
    tensor_out = torch.index_select(tensor_in, dim=0, index=idx_sort)
    label_out = torch.index_select(label_in, dim=0, index=idx_sort)
    return tensor_out, seq_len_out.squeeze(), label_out.squeeze(), idx_sort.squeeze()

if __name__ == '__main__':
    tensor_in = torch.tensor([
        [3, 1, 2, 0, 0, 0, 0, 0, 0, 0],
        [4, 34, 21, 6, 23, 1, 7, 3, 3, 8],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    seq_len = torch.tensor([
        [3],
        [10],
        [1]
    ])
    label_in = torch.tensor([
        [0],
        [7],
        [2]
    ])
    tensor_out, seq_len_out, label_out, index_s = tensor_seq_len_desent(tensor_in, seq_len, label_in)
    print(tensor_out, seq_len_out, label_out, index_s)
    source_label = torch.index_select(label_out, dim=0, index=index_s)
    print(source_label)