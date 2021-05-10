import sys
sys.path.append("../")

from eval_infer_base import EvalInfer

from utils.load import load_yaml, load_xlsx, load_json

class EvalInferFire(EvalInfer):
    """继承自infer_base.Infer基类，

    私有变量：self.configs, self.model_path, self.eval_data_path, self.label2index_json_path
    重写方法：self.predict()
    """
    def __init__(self, configs):
        super(EvalInferFire, self).__init__(configs)

    def predict(self):
        eval_df = load_xlsx(self.eval_data_file)
        data_x = list(eval_df["text"])
        data_y = list(eval_df["intent"])

        d = load_json(self.label2index_json_path)

        data_x_clear, data_y_clear = list(), list()
        for index in range(len(data_x)):
            if data_y[index] != "[]" and "," not in data_y[index] and data_y[index] in d.keys():
                data_x_clear.append(data_x[index])
                data_y_clear.append(data_y[index])
        # print(len(data_x_clear))
        # return data_x_clear, data_y_clear
        


if __name__=="__main__":

    configs = load_yaml("eval_infer_config.yaml")
    evalInferFire = EvalInferFire(configs)
    print(len(evalInferFire.predict()[0]))