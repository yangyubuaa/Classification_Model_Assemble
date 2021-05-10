import sys
sys.path.append("../../")
import torch

from tqdm import tqdm

from eval_infer_base import EvalInfer

from data_preprocess.Dataset.dataset import SequenceDataset, BertSequenceDataset


from utils.load import load_yaml, load_xlsx, load_json

class EvalInferFire(EvalInfer):
    """继承自infer_base.Infer基类，

    继承父类私有变量：
            self.configs 设置参数加载, 
            self.model_path 模型路径加载, 
            self.eval_data_path 需要测试的文件路径, 
            self.label2index_json_path 标签到索引的映射字典路径, 
            self.token2index_json_path 符号到索引的映射字典路径, 
            self.tokenizer 符号到token的转换工具,
            self.label_tokenizer 标签到索引的转换工具，
            self.model pytorch模型

    重写方法：
            self.predict()
            self.predict_one()

    """
    def __init__(self, configs):
        super(EvalInferFire, self).__init__(configs)

    def predict(self):
        
        eval_df = load_xlsx(self.eval_data_file)
        data_x = list(eval_df["text"])
        data_y = list(eval_df["intent"])

        d = load_json(self.label2index_json_path)

        # 加载干净的数据集
        data_x_clear, data_y_clear = list(), list()
        for index in range(len(data_x)):
            if data_y[index] != "[]" and "," not in data_y[index] and data_y[index] in d.keys():
                data_x_clear.append(data_x[index])
                data_y_clear.append(data_y[index])
        # print(len(data_x_clear))
        # return data_x_clear, data_y_clear

        predict_result = list()
        for index in tqdm(range(len(data_x_clear))):
            predict_result.append(self.predict_one(data_x_clear[index], data_y_clear[index]))
     
                
    def predict_one(self, input_sequence, label):
        """预测一条数据

        params: input_sequence: 一条语句
        params: label: 语句对应的正确标签

        return: predict_label
        """
        # 使用传统模型
        if self.model_name in ["fasttext", "lstm_base", "lstm_pack", "textcnn"]:
            print(self.tokenizer(input_sequence))
            raise Exception("未完成！")
        # 使用预训练模型
        else:
            tokenized = self.tokenizer(input_sequence, return_tensors="pt")
            # print(self.tokenizer(input_sequence, return_tensors="pt"))
            input_ids, token_type_ids, attention_mask = tokenized["input_ids"], tokenized["token_type_ids"], tokenized["attention_mask"]
            input_ids, token_type_ids, attention_mask = input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda()
            predict_result = self.model(input_ids, token_type_ids, attention_mask)
            # print(predict_result.shape)
            label = torch.argmax(predict_result, 0).cpu()
            # print(self.label_tokenizer.decoßde(label))

            
if __name__=="__main__":
    configs = load_yaml("eval_infer_config.yaml")
    evalInferFire = EvalInferFire(configs)
    evalInferFire.predict()