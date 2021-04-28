# -*- utf-8 -*-
# @Time: 2021/4/19 5:39 下午
# @Author: yang yu
# @File: train.py.py
# @Software: PyCharm
import torch

from utils.load import load_yaml
from dataset.preprocess import Preprocess
from data_preprocess.Tokenizer.tokenizer import SequenceTokenizer, ClassificationLabelTokenizer
from data_preprocess.Dataset.dataset import SequenceDataset
from data_preprocess.Cross_validation.cross_validation import KFoldCrossValidation

from torch.utils.data import DataLoader

from simple_classification.lstm_base.model import LSTM_Classfication

from torch.optim import Adam

from torch.nn.functional import cross_entropy

def train():
    use_cuda = True if torch.cuda.is_available() else False

    params = load_yaml("/Users/yangyu/PycharmProjects/infer_of_intent/dataset/preprocess_config.yaml")
    p = Preprocess(params)
    source_dataset_x, source_dataset_y = p.get_preprocessed_data()

    sequencetokenizer = SequenceTokenizer("/Users/yangyu/PycharmProjects/infer_of_intent/data_preprocess/vocab2index.json")
    classificationlabeltokenizer = ClassificationLabelTokenizer("/Users/yangyu/PycharmProjects/infer_of_intent/data_preprocess/label2index.json")

    (x_tokenized, x_lengths),  y_tokenized = sequencetokenizer(source_dataset_x), classificationlabeltokenizer(source_dataset_y)
    # print(x_tokenized.shape, y_tokenized.shape)
    dataset = SequenceDataset(x_tokenized, y_tokenized, x_lengths)

    # 使用交叉验证创建多个数据集，并将数据集切分为训练集和测试集，如果参数为0，那么作用为将原始数据集划分为训练集和测试集
    kFoldCV = KFoldCrossValidation(0)  # 10折交叉验证
    dataset_generator = kFoldCV(dataset, shuffle=False)  # 交叉验证类返回生成器，生成器每次返回一个交叉验证数据集

    for train, eval in dataset_generator:
        print(train, eval)

        model_params = load_yaml("/Users/yangyu/PycharmProjects/infer_of_intent/simple_classification/lstm_base/lstm_base_config.yaml")
        model = LSTM_Classfication(model_params)

        if use_cuda:
            model = model.cuda()

        optimizer = Adam(params=model.parameters(), lr=0.0001)

        epochs = model_params["epoch"]
        batch_size = model_params["batch_size"]
        train_dataloader = DataLoader(dataset=train, batch_size=64, shuffle=True)

        eval_dataloader = DataLoader(dataset=eval, batch_size=64)

        for batch in eval_dataloader:
            eval_x, eval_y, _ = batch
            eval_y = eval_y.squeeze()
            if use_cuda:
                eval_x, eval_y = eval_x.cuda(), eval_y.cuda()

        for epoch in range(epochs):
            for index, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                train_x, train_y, _ = batch
                train_y = train_y.squeeze()

                if use_cuda:
                    train_x, train_y = train_x.cuda(), train_y.cuda()

                # print(train_x.shape, train_y.shape)
                train_y_head = model(train_x)
                train_loss = cross_entropy(train_y_head, train_y)

                eval_y_head = model(eval_x)
                eval_loss = cross_entropy(eval_y_head, eval_y)

                train_predict = torch.argmax(train_y_head, 1)
                eval_predict = torch.argmax(eval_y_head, 1)
                train_accu = int((train_y == train_predict).sum()) / len(train_x)
                eval_accu = int((eval_y == eval_predict).sum()) / len(eval_x)

                train_loss.backward()
                optimizer.step()
                print("train_epoch:{} | train_batch:{} | train_loss:{} | eval_loss:{} | train_accu:{} | eval_accu:{}"
                      "".format(epoch, index, train_loss.item(), eval_loss.item(), train_accu, eval_accu))


if __name__ == '__main__':
    train()