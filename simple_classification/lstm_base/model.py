# -*- utf-8 -*-
# @Time: 2021/4/19 5:22 下午
# @Author: yang yu
# @File: model.py.py
# @Software: PyCharm

import torch

from utils.load import load_yaml

class LSTM_Classfication(torch.nn.Module):
    def __init__(self, configs: dict):
        super(LSTM_Classfication, self).__init__()
        self.configs = configs
        self.input_size = self.configs["nn_params"]["input_size"]
        self.embedding_size = self.configs["nn_params"]["embedding_size"]
        self.lstm_hiddensize = self.configs["nn_params"]["lstm_hiddensize"]
        self.bidirectional_lstm = self.configs["nn_params"]["bidirectional_lstm"]
        self.lstm_layers = self.configs["nn_params"]["lstm_layers"]
        self.drop_out = self.configs["nn_params"]["drop_out"]
        self.dense = self.configs["nn_params"]["dense"]
        self.label_nums = self.configs["nn_params"]["label_nums"]

        self.embedding_and_lstm = torch.nn.Sequential(
            torch.nn.Embedding(self.input_size, self.embedding_size),
            torch.nn.LSTM(input_size=self.embedding_size,
                          hidden_size=self.lstm_hiddensize,
                          bidirectional=self.bidirectional_lstm,
                          num_layers=self.lstm_layers,
                          dropout=self.drop_out)
        )
        # output shape of lstm: (seq_len, batch_size, lstm_hiddensize)
        self.classification = torch.nn.Sequential(
            torch.nn.Linear(self.lstm_hiddensize*2, self.dense),
            torch.nn.Linear(self.dense, self.label_nums)
        )

    def forward(self, input_batch):
        input_batch = input_batch.permute(1, 0)
        # print(input_batch.shape)
        lstm_out, _ = self.embedding_and_lstm(input_batch)
        # print(lstm_out.shape)
        return self.classification(lstm_out[-1, :, :].squeeze())

if __name__ == '__main__':
    model_params = load_yaml("/Users/yangyu/PycharmProjects/infer_of_intent/simple_classification/lstm_base/lstm_base_config.yaml")
    model = LSTM_Classfication(model_params)
