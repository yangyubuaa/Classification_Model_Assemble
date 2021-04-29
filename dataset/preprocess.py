# -*- utf-8 -*-
# @Time: 2021/4/19 10:11 上午
# @Author: yang yu
# @File: preprocess.py.py
# @Software: PyCharm

import pandas as pd
import random
import json
import os

from utils.load import load_yaml, load_xlsx, load_json


class Preprocess:
    '''将数据集封装，整理成训练集和测试集并返回，并产生label2index.json以及vocab2index.json

    '''
    def __init__(self, configs: dict):
        self.configs = configs

        self.dataset_path = self.configs["data_path"]["train"]
        self.eval_path = self.configs["data_path"]["eval"]

        self.vocab2index_json_path = self.configs["tokenized_path"]["vocab2index_json_path"]
        self.label2index_json_path = self.configs["tokenized_path"]["label2index_json_path"]

    def get_train_data(self):
        '''读取配置中的数据文件进行处理

        args: None
        return: PreprocessedData
        '''
        data_all_dataframe = load_xlsx(self.dataset_path)
        data_x = list(data_all_dataframe["text"])
        data_y = list(data_all_dataframe["intent"])

        # 遍历x，产生字符索引，注意，使用自定义的输入时存在OVV问题，需要添加[UNK]符号，该符号对应的索引设置为0
        vocabs = set()
        for index in range((len(data_x))):
            for w in data_x[index]:
                vocabs.add(w)

        labels = set(data_y)

        label_and_nums = [(data_y.count(label), label) for label in labels]
        print(sorted(label_and_nums))
        # label_need_replaced = [label for label in labels if data_y.count(label) < 20]
        # for index, label in enumerate(data_y):
        #     if data_y[index] in label_need_replaced:
        #         data_y[index] = "[]"
        # labels =  set(data_y)
        # print(len(vocabs), len(labels))

        vocab2index = {vocab: index+2 for index, vocab in enumerate(vocabs)}
        vocab2index["[PAD]"] = 0
        vocab2index["[UNK]"] = 1

        label2index = {label: index for index, label in enumerate(labels)}

        if not os.path.exists(self.vocab2index_json_path):
            with open(self.vocab2index_json_path, "w", encoding="utf-8") as vw:
                json.dump(vocab2index, vw, ensure_ascii=False, indent=2)
        if not os.path.exists(self.label2index_json_path):
            with open(self.label2index_json_path, "w", encoding="utf-8") as lw:
                json.dump(label2index, lw, ensure_ascii=False, indent=2)

        return data_x, data_y

    def get_eval_data(self):
        data_all_dataframe = load_xlsx(self.eval_path)
        data_x = list(data_all_dataframe["text"])
        data_y = list(data_all_dataframe["intent"])

        # for index in range(len(data_x) - 1, -1, -1):
        #     # 将多标签的数据替换为第一个标签
        #     if "," in data_y[index]:
        #         label_split = data_y[index].split(",")
        #         label = label_split[0][2:-1]
        #         data_y[index] = "['" + label + "']"

        d = load_json(self.label2index_json_path)

        data_x_clear, data_y_clear = list(), list()
        for index in range(len(data_x)):
            if data_y[index] != "[]" and "," not in data_y[index] and data_y[index] in d.keys():
                data_x_clear.append(data_x[index])
                data_y_clear.append(data_y[index])
        # print(len(data_x_clear))
        return data_x_clear, data_y_clear


if __name__ == '__main__':
    params = load_yaml("/Users/yangyu/PycharmProjects/infer_of_intent/dataset/preprocess_config.yaml")
    p = Preprocess(params)
    p.get_train_data()
    p.get_eval_data()
