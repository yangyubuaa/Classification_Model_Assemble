# -*- utf-8 -*-
# @Time: 2021/4/20 9:21 上午
# @Author: yang yu
# @File: tokenizer.py.py
# @Software: PyCharm
'''
序列的tokenizer以及分类label的tokenizer，在做tokenizer之后可以进一步封装为dataset类

'''
from utils.load import load_json

import torch

from utils.load import load_json

class SequenceTokenizer:
    def __init__(self, vocab2index: dict):
        '''初始化序列tokenizer类
        :param vocab2index: 参数可以直接传入tokenizer字典，也可以传入vocab2index路径
        '''
        if isinstance(vocab2index, dict):
            self.vocab2index = vocab2index
        else:
            self.vocab2index = load_json(vocab2index)
        self.index2vocab = {value: key for key, value in self.vocab2index.items()}

    def __call__(self, input_sequence: list, r_tensor=True, padding_max=True, padding_value=0, cut_sentence=False, cut_thres=0):
        '''transfer token to number

        当序列为batch时，会返回数据tensor以及长度tensor;
        当序列为一个sentence时，会返回tensor

        :param input_sequence: list():one sample or list(list(), ..., list()):batch sample
        :param r_tensor: 默认为True，返回tokenized tensor
        :param padding_max: 默认为True，按照序列最长填充
        :param padding_value: 默认为填充0
        :param cut_sentence: 默认为False
        :param cut_thres: 当cut_sentence为True时起作用
        :return:
        '''

        # batch sample
        if isinstance(input_sequence, list):

            num_sequence = [list() for i in range(len(input_sequence))]
            length_of_sequence = [len(i) for i in input_sequence]

            for i, sentence in enumerate(input_sequence):
                for j, v in enumerate(input_sequence[i]):
                    try:
                        num_sequence[i].append(self.vocab2index[input_sequence[i][j]])
                    except:
                        num_sequence[i].append(self.vocab2index["[UNK]"])

            if padding_max and cut_sentence:
                raise Exception("can't padding_max and cut_sentence in the same time!")

            if padding_max:
                max_length = len(max(input_sequence, key=len))
                for i in range(len(num_sequence)):
                    if len(num_sequence[i]) < max_length:
                        for j in range(max_length-len(num_sequence[i])):
                            num_sequence[i].append(padding_value)
                if r_tensor:
                    return torch.tensor(num_sequence), torch.tensor(length_of_sequence).unsqueeze(-1)
                else:
                    return num_sequence, length_of_sequence

            if cut_sentence:
                raise Exception("cut_sentence has not been completed!")
        # one sample
        else:
            try:
                nums = [self.vocab2index[i] for i in input_sequence]
            except:
                nums = [self.vocab2index["[UNK]"] for i in input_sequence]
            if r_tensor:
                return torch.tensor(nums)
            else:
                return nums

    def decode(self, tokenized_tensor, length_tensor):
        # 存在问题，oov词汇decode不出来，未解决bug
        # shape of tokenized_tensor is: (batch_size, max_seq_len)
        # shape of length_tensor is: (batch_size, 1)
        tokenized_list = tokenized_tensor.numpy().tolist()
        length_of_sequence = length_tensor.squeeze().numpy().tolist()
        # print(tokenized_list, length_of_sequence)

        for index in range(len(tokenized_list)):
            tokenized_list[index] = tokenized_list[index][:length_of_sequence[index]]

        # print(tokenized_list)
        for i in range(len(tokenized_list)):
            for j in range(len(tokenized_list[i])):
                tokenized_list[i][j] = self.index2vocab[tokenized_list[i][j]]

        for i in range(len(tokenized_list)):
            tokenized_list[i] = "".join(tokenized_list[i])

        return tokenized_list

class ClassificationLabelTokenizer:
    def __init__(self, label2index: dict):
        if isinstance(label2index, dict):
            self.label2index = label2index
        else:
            self.label2index = load_json(label2index)
        self.index2label = {value:key for key, value in self.label2index.items()}

    def __call__(self, input_label, r_tensor=True):
        if isinstance(input_label, list):
            num_label = [self.label2index[i] for i in input_label]
            if r_tensor:
                return torch.tensor(num_label).unsqueeze(-1)
            else:
                return num_label
        else:
            num_label = self.label2index[input_label]
            if r_tensor:
                return torch.tensor(num_label).unsqueeze(-1)
            else:
                return num_label

    def decode(self, label_tensor):
        label_list = label_tensor.squeeze().numpy().tolist()
        if isinstance(label_list, int):
            return [self.index2label[label_list]]
        else:
            for index in range(len(label_list)):
                label_list[index] = self.index2label[label_list[index]]

            return label_list


if __name__ == '__main__':
    # SequenceTokenizer使用方法1
    vocab2index = load_json("/Users/yangyu/PycharmProjects/infer_of_intent/data_preprocess/vocab2index.json")
    sequencetokenizer = SequenceTokenizer(vocab2index)
    tokenized = sequencetokenizer(["视网膜脱离手术并发症",
                                   "视网膜脱离手术",
                                   "视网膜"])

    # SequenceTokenizer使用方法2
    sequencetokenizer = SequenceTokenizer("/Users/yangyu/PycharmProjects/infer_of_intent/data_preprocess/vocab2index.json")
    tokenized = sequencetokenizer(["视网膜脱离手术并发症",
                                   "视网膜脱离手术",
                                   "视网膜"])

    # ClassificationLabelTokenizer使用方法1
    classificationlabeltokenizer = ClassificationLabelTokenizer("/Users/yangyu/PycharmProjects/infer_of_intent/data_preprocess/label2index.json")
    tokenized = classificationlabeltokenizer(["['护理_疾病护理']", "['护理_疾病护理']"])
    print(tokenized)
