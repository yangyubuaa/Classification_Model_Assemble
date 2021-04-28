import torch
from torch.utils.data import Dataset
from random import shuffle
'''
在做tokenize之后使用下面的类进行数据加载
'''


class SequenceDataset(Dataset):
    '''序列分类数据集类，继承自Dataset

    1) 实现shuffle_()函数可以将数据进行打乱，在原本的对象上进行操作
    2) 实现extend_()函数可以将同类数据集进行扩展，在原本的对象上进行操作
    3) 实现slice()函数实现，返回一个新的切片对象
    '''
    def __init__(self, data_x, data_y, seq_length):
        self.data_x = data_x  # tensor, shape(data_size, seq_len)
        self.data_y = data_y  # tensor, shape(data_size, 1)
        self.sequence_length = seq_length  # tensor, shape(data_size, 1)
        try:
            assert torch.is_tensor(self.data_x)
            assert torch.is_tensor(self.data_y)
            assert torch.is_tensor(self.sequence_length)
            assert self.data_y.shape == (len(self.data_x), 1)
            assert self.sequence_length.shape == (len(self.data_x), 1)
        except:
            raise Exception("input data is illegal!Please check the data type and data shape!")

    def __getitem__(self, item):
        '''
        此处要实现torch dataset类的效果，切片不在此实现，提供一个新的方法实现slice()切片
        :param item:
        :return:
        '''
        return self.data_x[item], self.data_y[item], self.sequence_length[item]

    def __len__(self):
        return len(self.data_x)

    def __repr__(self):
        '''
        检查类的内部信息
        :return:
        '''
        data_x_shape = self.data_x.shape
        data_y_shape = self.data_y.shape
        sequence_length_shape = self.sequence_length.shape
        # print(data_x_shape, data_y_shape, sequence_length_shape)
        return 'shape of data_x({},{}),data_y({},{}),sequence_length({},{})'.format(
            data_x_shape[0], data_x_shape[1], data_y_shape[0], data_y_shape[1], sequence_length_shape[0], sequence_length_shape[1]
        )

    def shuffle_(self):
        '''将数据打乱
        :return:
        '''
        new_data_x, new_data_y, new_seq_length = list(), list(), list()
        # print(len(self.data_x))
        source_index = [i for i in range(len(self.data_x))]
        shuffle(source_index)
        shuffle_index = source_index
        for i in shuffle_index:
            new_data_x.append(list(self.data_x[i]))
            new_data_y.append(int(self.data_y[i]))
            new_seq_length.append(int(self.sequence_length[i]))
        # print(new_data_x)
        self.data_x = torch.tensor(new_data_x)
        self.data_y = torch.tensor(new_data_y).unsqueeze(-1)
        self.sequence_length = torch.tensor(new_seq_length).unsqueeze(-1)

    def extend_(self, other):
        try:
            assert isinstance(other, SequenceDataset)
        except:
            raise Exception("Please input SequenceDataset to extend!")

        try:
            assert self.data_x.shape[1] == other.data_x.shape[1]
            assert self.data_y.shape[1] == other.data_y.shape[1]
            assert self.sequence_length.shape[1] == other.sequence_length.shape[1]
        except:
            raise Exception("The shape of other tensor is wrong!")
        self.data_x = torch.cat((self.data_x, other.data_x), 0)
        self.data_y = torch.cat((self.data_y, other.data_y), 0)
        self.sequence_length = torch.cat((self.sequence_length, other.sequence_length), 0)

    def slice(self, start, end):
        return SequenceDataset(self.data_x[start:end], self.data_y[start:end], self.sequence_length[start:end])


class BertSequenceDataset(Dataset):
    def __init__(self, data_x_bert_tokenized: dict, data_y):
        self.data_x_bert_tokenized = data_x_bert_tokenized  # dict{input_ids:tensor}
        self.data_y = data_y  # (batch_size, 1)

    def __getitem__(self, item):
        return self.data_x_bert_tokenized["input_ids"][item],\
               self.data_x_bert_tokenized["token_type_ids"][item],\
               self.data_x_bert_tokenized["attention_mask"][item],\
               self.data_y[item]

    def __len__(self):
        return len(self.data_y)


if __name__ == '__main__':
    pass
    # 测试
    # data_x = torch.tensor(
    #     [
    #         [11, 5, 3, 9, 6, 1, 3, 4, 10, 9],
    #         [23, 1, 4, 32, 42, 32, 21, 3, 9, 9],
    #         [22, 3, 1, 4, 5, 89, 76, 6, 8, 3],
    #         [21, 87, 45, 34, 3, 0, 0, 0, 0, 0],
    #         [3, 1, 3, 4, 5, 0, 0, 0, 0, 0],
    #         [23, 12, 3, 0, 0, 0, 0, 0, 0, 0]
    #     ]
    # )
    # data_y = torch.tensor(
    #     [
    #         [1],
    #         [5],
    #         [2],
    #         [7],
    #         [9],
    #         [3]
    #     ]
    # )
    # sequence_length = torch.tensor(
    #     [
    #         [10],
    #         [10],
    #         [10],
    #         [5],
    #         [5],
    #         [3]
    #     ]
    # )
    # s = SequenceDataset(data_x, data_y, sequence_length)
    # s.shuffle_()
    # print(repr(s))
    # s.extend_(s)
    # print(repr(s))
    # a = s.slice(0, 2)
    # print(a)
    # a.shuffle_()
    # print(a)
    #

