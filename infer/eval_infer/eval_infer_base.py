import sys
sys.path.append("../")

import torch

from utils.load import load_yaml, load_xlsx, load_json
from transformers import BertTokenizer, ElectraTokenizer, XLNetTokenizer
from data_preprocess.Tokenizer.tokenizer import SequenceTokenizer
from data_preprocess.Tokenizer.label_tokenizer import ClassificationLabelTokenizer

# 评估推断方法基类（带标签）
class EvalInfer:
    """评估推断方法基类

    变量：
    self.configs
    self.model_name
    self.model_path
    self.eval_data_file
    self.label2index_json_path
    self.token2index_json_path
    self.tokenizer
    self.label_tokenizer
    self.model

    方法：
    predict()
    predict_one()
    """
    def __init__(self, configs: dict):
        
        # 加载参数字典
        self.configs = configs
        
        # 设置模型以及路径
        if self.configs["model_select"] == "fasttext":
            self.model_path = self.configs["model_path"]["fasttext"]
        elif self.configs["model_select"] == "lstm_base":
            self.model_path = self.configs["model_path"]["lstm_base"]
        elif self.configs["model_select"] == "lstm_pack":
            self.model_path = self.configs["model_path"]["lstm_pack"]
        elif self.configs["model_select"] == "textcnn":
            self.model_path = self.configs["model_path"]["textcnn"]
        elif self.configs["model_select"] == "bert":
            self.model_path = self.configs["model_path"]["bert"]
        elif self.configs["model_select"] == "electra":
            self.model_path = self.configs["model_path"]["electra"]
        elif self.configs["model_select"] == "xlnet":
            self.model_path = self.configs["model_path"]["xlnet"]

        # 设置模型名称
        self.model_name = self.configs["model_select"]

        # 设置评估推断的数据文件
        self.eval_data_file = self.configs["eval_data_file"]

        # 设置标签转换路径，需要将预测的label转为真实标签
        self.label2index_json_path = self.configs["eval_label_transfer_file"]

        # 设置token映射词表，在自定义模型中的tokenizer使用
        self.token2index_json_path = self.configs["eval_token_transfer_file"]

        # 设置分词器
        if self.model_name in ["fasttext", "lstm_base", "lstm_pack", "textcnn"]:
            self.tokenizer = SequenceTokenizer(load_json(self.token2index_json_path))

        elif self.model_name == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(self.configs["pretrained_model_path"]["bert"])
        elif self.model_name == "electra":
            self.tokenizer = ElectraTokenizer.from_pretrained(self.configs["pretrained_model_path"]["electra"])
        elif self.model_name == "xlnet":
            self.tokenizer = XLNetTokenizer.from_pretrained(self.configs["pretrained_model_path"]["xlnet"])

        # 设置label转换器
        self.label_tokenizer = ClassificationLabelTokenizer(load_json(self.label2index_json_path))

        # 加载模型
        self.model = torch.load(self.model_path)

    # 预测方法（预测整个数据集）
    def predict(self):
        pass

    # 预测一个句子
    def predict_one(self):
        pass





if __name__=="__main__":
    # 推断基类测试
    configs = load_yaml("eval_infer_config.yaml")
    evalInfer = EvalInfer(configs)