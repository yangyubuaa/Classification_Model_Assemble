import sys
sys.path.append("../")

import torch

from utils.load import load_yaml, load_xlsx

# 评估推断方法基类（带标签）
class EvalInfer:
    """评估推断方法基类
    """
    def __init__(self, configs: dict):
        
        # 设置字典
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

        # 设置评估推断的数据文件
        self.eval_data_file = self.configs["eval_data_file"]

        # 设置标签转换路径，需要将预测的label转为真实标签
        self.label2index_json_path = self.configs["eval_label_transfer_file"]

    # 预测方法
    def predict(self):
        pass





if __name__=="__main__":
    # 推断基类测试
    configs = load_yaml("eval_infer_config.yaml")
    evalInfer = EvalInfer(configs)