import torch
import pandas as pd

from simple_classification.lstm_pack.model import LSTM_Classfication_Packed
from data_preprocess.Tokenizer.tokenizer import SequenceTokenizer, ClassificationLabelTokenizer
from utils.load import load_json
from data_preprocess.Dataset.dataset import SequenceDataset
from data_preprocess.TensorSort.tensor_sort import tensor_seq_len_desent

def predict(model, sequencetokenizer, labeltokenizer, sentence):
    tokenized = sequencetokenizer(sentence)
    # print(tokenized)
    length = torch.tensor([len(tokenized)])
    tokenized = tokenized.unsqueeze(0)
    predict_y = model(tokenized, length)
    p = torch.argmax(predict_y)
    return labeltokenizer.decode(p)

def load_eval_predict():
    eval_set = pd.read_excel("/Users/yangyu/PycharmProjects/infer_of_intent/dataset/eval_data/eval.xlsx")
    eval_x, eval_y = list(eval_set["text"]), list(eval_set["intent"])

    eval_x_clear, eval_y_clear = list(), list()
    label2index = load_json("/Users/yangyu/PycharmProjects/infer_of_intent/dataset/label2index.json")
    # label有问题的样本
    eval_x_removed = list()
    # label在训练集中不存在的样本
    eval_x_not_exist_in_trainset = list()
    # 将label有问题的([]、多label)样本去除掉
    for index in range(len(eval_x)):
        # 单标签
        if "," not in eval_y[index] and eval_y[index]!="[]":
            # 单标签未在训练集中出现
            if eval_y[index] not in label2index.keys():
                eval_x_not_exist_in_trainset.append(eval_x[index])
            # 满足测试条件的测试数据
            else:
                eval_x_clear.append(eval_x[index])
                eval_y_clear.append(eval_y[index])
        # 多标签或者问题标签
        else:
            eval_x_removed.append(eval_x[index])

    print(len(eval_x), len(eval_x_clear), len(eval_x_removed), len(eval_x_not_exist_in_trainset))
    # 4708 4222 481 5

    # eval_x：全部测试样本，eval_x_clear：进行测试的样本，eval_x_removed：标签问题去除的样本，eval_x_not_exist_in_trainset：标签不在训练集的去除样本

    model = torch.load("model.bin")
    # model = model.eval()
    sequencetokenizer = SequenceTokenizer("/Users/yangyu/PycharmProjects/infer_of_intent/dataset/vocab2index.json")
    labeltokenizer = ClassificationLabelTokenizer("/Users/yangyu/PycharmProjects/infer_of_intent/dataset/label2index.json")

    saved_x, saved_label, saved_predict_true, saved_predict_result, label_wrong, label_not_in_train_set = list(), list(), list(), list(), list(), list()

    for index in range(len(eval_x)):
        if eval_x[index] in eval_x_clear:
            saved_x.append(eval_x[index])
            saved_label.append(eval_y[index])
            p = predict(model, sequencetokenizer, labeltokenizer, eval_x[index])
            print(p[0])
            if p[0] == eval_y[index]:
                saved_predict_true.append(1)
            else:
                saved_predict_true.append(0)

            saved_predict_result.append(p)
            label_wrong.append("NA")
            label_not_in_train_set.append("NA")

        if eval_x[index] in eval_x_removed:
            saved_x.append(eval_x[index])
            saved_label.append(eval_y[index])
            saved_predict_true.append("NA")
            saved_predict_result.append("NA")
            label_wrong.append("TRUE")
            label_not_in_train_set.append("NA")

        if eval_x[index] in eval_x_not_exist_in_trainset:
            saved_x.append(eval_x[index])
            saved_label.append(eval_y[index])
            saved_predict_true.append("NA")
            saved_predict_result.append("NA")
            label_wrong.append("NA")
            label_not_in_train_set.append("TRUE")


    print(len(saved_x), len(saved_label), len(saved_predict_true), len(saved_predict_result), len(label_wrong), len(label_not_in_train_set))
    predict_df = pd.DataFrame(
        {
            "text":saved_x,
            "intent":saved_label,
            "p_true/false":saved_predict_true,
            "p_intent":saved_predict_result,
            "label_wrong":label_wrong,
            "not_in_train_set":label_not_in_train_set
        }
    )
    predict_df.to_excel("predict_result.xlsx")

    print(saved_predict_true.count(1) / len(eval_x_clear))

if __name__ == '__main__':
    load_eval_predict()