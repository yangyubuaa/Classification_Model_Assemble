import torch
import math
import sys
sys.path.append("../")
import pandas as pd

from utils.load import load_xlsx, load_json, load_yaml

class MultilabelF1_Score:
    def __init__(self, configs):
        
        self.configs = configs
        self.predicted_dataset_path = self.configs["predicted_dataset_path"]
        self.label2index_json_path = self.configs["label2index_json_path"]

        # dataframe
        self.eval_data = load_xlsx(self.predicted_dataset_path)
        # dict，可视化时将索引转为label
        self.label2index = load_json(self.label2index_json_path)
        self.index2label = {value: key for key, value in self.label2index.items()}

        # confusion_matrix
        self.confusion_matrix = torch.zeros((len(self.label2index), len(self.label2index)))
        # 初始化混淆矩阵
        self.confusion_matrix_init()
        # 记录测试集中不存在的类别（list）
        self.not_in_test = self.confusion_matrix_clear()

    def confusion_matrix_init(self):
        """根据任务所有类别初始化混淆矩阵
        """
        # print(self.confusion_matrix)
        true_label = list(self.eval_data["intent"])
        predict_label = list(self.eval_data["predict"])
        for index in range(len(predict_label)):
            predict_label[index] = predict_label[index][2:-2]
            # print(true_label[index], predict_label[index])
        # print(len(true_label), len(predict_label))

        for index in range(len(true_label)):
            self.confusion_matrix[self.label2index[true_label[index]]][self.label2index[predict_label[index]]] = \
            self.confusion_matrix[self.label2index[true_label[index]]][self.label2index[predict_label[index]]] + 1

    def confusion_matrix_clear(self):
        """记录测试集中不存在的类别
        召回率可能存在为nan的情况，表明该索引对应的类别在测试集中不存在，需要将混淆矩阵中的该类别去掉
        """
        not_in_test = list()
        sums = torch.sum(self.confusion_matrix, dim=1)
        for i in range(len(sums)):
            if sums[i].item()==0:
                not_in_test.append(i)
        
        return not_in_test
        

    def precision(self):
        """根据混淆矩阵计算精度
        """
        precisions = dict()
        sums = torch.sum(self.confusion_matrix, dim=0)
        for index in range(len(self.confusion_matrix)):
            if index not in self.not_in_test:
                precisions[index] = self.confusion_matrix[index][index] / sums[index]

        for key in precisions.keys():
            precisions[key] = precisions[key].item()

        return precisions

    def recall(self):
        """根据混淆矩阵计算召回率
        """
        recalls = dict()
        sums = torch.sum(self.confusion_matrix, dim=1)
        for index in range(len(self.confusion_matrix)):
            if index not in self.not_in_test:
                recalls[index] = self.confusion_matrix[index][index] / sums[index]
        
        for key in recalls.keys():
            recalls[key] = recalls[key].item()
        
        return recalls
    
    def f1_score(self):
        precisions = self.precision()
        # print(precisions)
        recalls = self.recall()
        print(recalls)
        f1_scores = dict()
        for key in precisions.keys():
            p = precisions[key]
            r = recalls[key]
            print(p, r)
            # 会存在精度和召回率都为0的情况
            if p+r != 0:
                f1 = 2 * p * r / (p + r)
            else:
                f1 = 0 
            print(f1)
            f1_scores[key] = f1

        return f1_scores
    
    def visualize(self):
        """将精度、召回率、f1_score存储
        """
        precisions = self.precision()
        recalls = self.recall()
        f1_scores = self.f1_score()

        accumulates = dict()

        for key in precisions.keys():
            accumulates[self.index2label[key]] = [precisions[key], recalls[key], f1_scores[key]]

        print(accumulates)
        df = pd.DataFrame(accumulates)
        df.to_excel("score.xlsx")

        # 打印混淆矩阵
        df = pd.DataFrame(self.confusion_matrix.numpy())
        df.index = self.label2index.keys()
        df.columns = self.label2index.keys()
        print(df)
        for i in self.not_in_test:
            df = df.drop(self.index2label[i], axis=0)
            df = df.drop(self.index2label[i], axis=1)
        df.to_excel("confusion_matrix.xlsx")



if __name__=="__main__":
    configs = load_yaml("/home/ubuntu1804/pytorch_sequence_classification/algorithm_assess/assess_config.yaml")
    m = MultilabelF1_Score(configs)
    m.precision()
    m.recall()
    print(m.f1_score())
    m.visualize()