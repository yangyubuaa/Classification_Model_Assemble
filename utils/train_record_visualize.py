import sys
sys.path.append("../")
from utils.load import load_json

import matplotlib.pyplot as plt
# import seaborn as sns

class TrainRecordVisualize:
    """训练过程数据可视化，1）实现单模型的可视化，2）实现多模型的对比
    """
    def __init__(self, train_record_json):

        # 加载训练过程中产生的数据
        self.train_params = load_json(train_record_json)
        

    def visualize(self):
        print(self.train_params)

        fig_x = list()
        train_loss = list()
        eval_loss = list()
        train_accu = list()
        eval_accu = list()

        for value in self.train_params["train_record"].values():
            fig_x.append(str(value[0])+"-"+str(value[1]))
            train_loss.append(value[2])
            eval_loss.append(value[3])
            train_accu.append(value[4])
            eval_accu.append(value[5])

        x = range(len(fig_x))
        plt.plot(fig_x, train_loss, "r--", label="train_loss")
        plt.plot(fig_x, eval_loss, "g--", label="eval_loss")
        plt.plot(fig_x, train_accu, "b--", label="train_accu")
        plt.plot(fig_x, eval_accu, "y--", label="eval_accu")
        plt.xticks(x, fig_x)
        plt.savefig('testblueline.jpg')

        raise Exception("未完成！")



if __name__ == "__main__":
    t = TrainRecordVisualize("/home/ubuntu1804/pytorch_sequence_classification/transformers_based_classification/electra_classification/train_recordsave_params.json")
    t.visualize()