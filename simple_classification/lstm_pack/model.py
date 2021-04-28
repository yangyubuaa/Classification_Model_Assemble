# -*- utf-8 -*-
# @Time: 2021/4/20 3:00 下午
# @Author: yang yu
# @File: model.py.py
# @Software: PyCharm

# Lstm分类的实现，为了使用统一的训练接口，将数据集操作放在模型内部

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.load import load_yaml
from data_preprocess.TensorSort.tensor_sort import tensor_seq_len_desent

class LSTM_Classfication_Packed(torch.nn.Module):
    def __init__(self, configs: dict):
        super(LSTM_Classfication_Packed, self).__init__()
        self.configs = configs
        self.input_size = self.configs["nn_params"]["input_size"]
        self.embedding_size = self.configs["nn_params"]["embedding_size"]
        self.lstm_hiddensize = self.configs["nn_params"]["lstm_hiddensize"]
        self.bidirectional_lstm = self.configs["nn_params"]["bidirectional_lstm"]
        self.lstm_layers = self.configs["nn_params"]["lstm_layers"]
        self.drop_out = self.configs["nn_params"]["drop_out"]
        self.dense = self.configs["nn_params"]["dense"]
        self.label_nums = self.configs["nn_params"]["label_nums"]

        self.embedding = torch.nn.Sequential(
            torch.nn.Embedding(self.input_size, self.embedding_size),
        )

        self.lstm = torch.nn.Sequential(
            torch.nn.LSTM(input_size=self.embedding_size,
                          hidden_size=self.lstm_hiddensize,
                          bidirectional=self.bidirectional_lstm,
                          num_layers=self.lstm_layers,
                          dropout=self.drop_out,
                          batch_first=True)
        )

        # output shape of lstm: (seq_len, batch_size, lstm_hiddensize)
        self.classification = torch.nn.Sequential(
            torch.nn.Linear(self.lstm_hiddensize*2, self.dense),
            torch.nn.ReLU(),
            torch.nn.Linear(self.dense, self.label_nums)
        )

    def forward(self, input_batch, batch_seqs_len):
        embedded = self.embedding(input_batch)
        embedded_packed = pack_padded_sequence(embedded, batch_seqs_len, batch_first=True)
        # print(embedded_packed)
        lstm_out, _ = self.lstm(embedded_packed)
        lstm_out = pad_packed_sequence(lstm_out)[0]
        # print(lstm_out.shape)
        return self.classification(lstm_out[0, :, :].squeeze())

if __name__ == '__main__':
    params_dict = load_yaml("lstm_pack_config.yaml")
    LSTM_Classfication_Packed(params_dict)