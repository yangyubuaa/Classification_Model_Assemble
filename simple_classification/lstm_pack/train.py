# -*- utf-8 -*-
# @Time: 2021/4/21 10:15 上午
# @Author: yang yu
# @File: train.py.py
# @Software: PyCharm

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
from data_preprocess.TensorSort.tensor_sort import tensor_seq_len_desent

from torch.utils.data import DataLoader

from simple_classification.lstm_pack.model import LSTM_Classfication_Packed

from torch.optim import Adam

from torch.nn.functional import cross_entropy

from utils.train_record import TrainProcessRecord


def train():
    # 是否使用gpu加速
    use_cuda = True if torch.cuda.is_available() else False

    # 预处理参数读取
    params = load_yaml("/Users/yangyu/PycharmProjects/infer_of_intent/dataset/preprocess_config.yaml")
    p = Preprocess(params)
    # 得到原始数据集训练文本和标签
    source_train_x, source_train_y = p.get_train_data()
    source_eval_x, source_eval_y = p.get_eval_data()

    # 初始化数据和标签tokenizer
    sequencetokenizer = SequenceTokenizer(params["tokenized_path"]["vocab2index_json_path"])
    classificationlabeltokenizer = ClassificationLabelTokenizer(params["tokenized_path"]["label2index_json_path"])

    # tokenized
    (tokenized_train_x, tokenized_train_x_lengths), tokenized_train_y = sequencetokenizer(source_train_x), classificationlabeltokenizer(source_train_y)

    (tokenized_eval_x, tokenized_eval_x_lengths), tokenized_eval_y = sequencetokenizer(source_eval_x), classificationlabeltokenizer(source_eval_y)

    # 构建训练数据集
    train_dataset = SequenceDataset(tokenized_train_x, tokenized_train_y, tokenized_train_x_lengths)

    # 构建测试数据集
    eval_dataset = SequenceDataset(tokenized_eval_x, tokenized_eval_y, tokenized_eval_x_lengths)

    model_params = load_yaml("lstm_pack_config.yaml")
    model = LSTM_Classfication_Packed(model_params)

    train_record = TrainProcessRecord(1000, 50)
    params_s = []

    if use_cuda:
        model = model.cuda()

    optimizer = Adam(params=model.parameters(), lr=0.0001)

    epochs = model_params["epoch"]
    batch_size = model_params["batch_size"]
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=64)

    for epoch in range(epochs):
        for index, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            train_x, train_y, train_seqlen = batch
            train_y = train_y.squeeze()

            if use_cuda:
                train_x, train_y, train_seqlen = train_x.cuda(), train_y.cuda(), train_seqlen.cuda()

            train_y = train_y.unsqueeze(-1)
            # 对输入的tensor进行重排序
            # print(train_x.shape, train_y.shape, train_seqlen.shape)
            train_x, train_seqlen, train_y, _ = tensor_seq_len_desent(train_x, train_seqlen, train_y)

            # print(train_x.shape, train_y.shape)
            train_y_head = model(train_x, train_seqlen)
            train_loss = cross_entropy(train_y_head, train_y)

            train_loss.backward()
            optimizer.step()

            if index % 100 == 0:
                train_predict = torch.argmax(train_y_head, 1)
                train_accu = int((train_y == train_predict).sum()) / len(train_x)
                sum_eval_accu = 0
                sum_eval_loss = 0
                for batch in eval_dataloader:
                    eval_x, eval_y, eval_seqlen = batch
                    eval_y = eval_y.squeeze()
                    if use_cuda:
                        eval_x, eval_y, eval_seqlen = eval_x.cuda(), eval_y.cuda(), eval_seqlen.cuda()
                    eval_y = eval_y.unsqueeze(-1)
                    eval_x, eval_seqlen, eval_y, _ = tensor_seq_len_desent(eval_x, eval_seqlen, eval_y)

                    eval_y_head = model(eval_x, eval_seqlen)
                    eval_loss = cross_entropy(eval_y_head, eval_y)

                    eval_predict = torch.argmax(eval_y_head, 1)
                    eval_accu = int((eval_y == eval_predict).sum()) / len(eval_x)
                    sum_eval_accu = sum_eval_accu + eval_accu
                    sum_eval_loss = sum_eval_loss + eval_loss

                sum_eval_accu = sum_eval_accu / len(eval_dataloader)
                sum_eval_loss = sum_eval_loss / len(eval_dataloader)
                print("train_epoch:{} | train_batch:{} | train_loss:{} | eval_loss:{} | train_accu:{} | eval_accu:{}"
                      "".format(epoch, index, train_loss.item(), sum_eval_loss, train_accu, sum_eval_accu))
                # train_record(model, params_s, epoch, index, train_loss, eval_loss, train_accu, eval_accu)


if __name__ == '__main__':
    train()