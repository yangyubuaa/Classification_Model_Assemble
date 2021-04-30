import torch
import torch.nn as nn
import argparse
import sys
sys.path.append("../..")
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn.functional import cross_entropy

from transformers import BertTokenizer

from utils.load import load_yaml
from dataset.preprocess import Preprocess
from transformers_based_classification.bert_classification.model import BertClassification
from data_preprocess.Dataset.dataset import BertSequenceDataset
from data_preprocess.Tokenizer.label_tokenizer import ClassificationLabelTokenizer


def train():
    # 是否使用gpu加速
    use_cuda = True if torch.cuda.is_available() else False
    if use_cuda:
        device_nums = torch.cuda.device_count()
        print("use {} GPUs!".format(device_nums))

    # 预处理以及模型参数读取
    train_configs = load_yaml("bert_classification_config.yaml")
    params = load_yaml(train_configs["path"]["preprocess_config_path"])
    p = Preprocess(params)

    # 得到原始数据集训练文本和标签
    source_train_x, source_train_y = p.get_train_data()
    source_eval_x, source_eval_y = p.get_eval_data()

    # 初始化bert输入和label的tokenizer
    berttokenizer = BertTokenizer.from_pretrained(train_configs["path"]["bert_path"])
    labeltokenizer = ClassificationLabelTokenizer(params["tokenized_path"]["label2index_json_path"])

    # 将训练集和测试集进行tokenize
    train_x_tokenized = berttokenizer(source_train_x, padding=True, truncation=True, return_tensors="pt")
    train_y_tokenized = labeltokenizer(source_train_y)

    eval_x_tokenized = berttokenizer(source_eval_x, padding=True, truncation=True, return_tensors="pt")
    eval_y_tokenized = labeltokenizer(source_eval_y)

    # 创建训练集和测试集
    train_set = BertSequenceDataset(train_x_tokenized, train_y_tokenized)
    eval_set = BertSequenceDataset(eval_x_tokenized, eval_y_tokenized)

    # 读取模型参数初始化模型
    model = BertClassification(train_configs)
    if use_cuda:
        model = nn.DataParallel(model, device_ids=list(range(device_nums)))
        model = model.cuda(device=0)
    # 使用dataloader加载训练集和测试集
    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
    eval_dataloader = DataLoader(eval_set, batch_size=8)

    # 创建优化器
    optmizer = SGD(model.parameters(), lr=0.00001)

    # 创建损失函数
    loss_f = nn.CrossEntropyLoss()
    for epoch in range(100):
        for batch_index, batch in enumerate(train_dataloader):
            # print(batch_index)
            optmizer.zero_grad()
            input_ids, token_type_ids, attention_mask, train_y = batch
            # l = input_ids.numpy().tolist()
            # for i in l:
            #     print(berttokenizer.decode(i))
            # print(labeltokenizer.decode(train_y))
            if use_cuda:
                input_ids, token_type_ids, attention_mask, train_y = \
                    input_ids.cuda(device=0), token_type_ids.cuda(device=0), attention_mask.cuda(device=0), train_y.cuda(device=0)


            train_y_predict = model(input_ids, token_type_ids, attention_mask)
            train_y = train_y.squeeze()
            train_loss = loss_f(train_y_predict, train_y)

            train_loss.backward()
            optmizer.step()
            print(train_loss)
            if train_loss < 0.01 or epoch == 20:
                model = model.cpu()
                torch.save(model, "model.bin")
                return

            if batch_index % 100 == 0:
                train_predict = torch.argmax(train_y_predict, 1)
                train_accu = int((train_y == train_predict).sum()) / len(train_y)
                sum_eval_accu = 0
                sum_eval_loss = 0
                for eval_batch in eval_dataloader:
                    eval_input_ids, eval_token_type_ids, eval_attention_mask, eval_y = eval_batch
                    eval_y = eval_y.squeeze()
                    if use_cuda:
                        eval_input_ids, eval_token_type_ids, eval_attention_mask, eval_y = \
                            eval_input_ids.cuda(device=1), eval_token_type_ids.cuda(device=1), eval_attention_mask.cuda(device=1), eval_y.cuda(device=1)

                    eval_y_predict = model(eval_input_ids, eval_token_type_ids, eval_attention_mask)
                    eval_loss = cross_entropy(eval_y_predict, eval_y)

                    eval_predict = torch.argmax(eval_y_predict, 1)
                    eval_accu = int((eval_y == eval_predict).sum()) / len(eval_y)
                    sum_eval_accu = sum_eval_accu + eval_accu
                    sum_eval_loss = sum_eval_loss + eval_loss
                    optmizer.zero_grad()
                    torch.cuda.empty_cache()
                sum_eval_accu = sum_eval_accu / len(eval_dataloader)
                sum_eval_loss = sum_eval_loss / len(eval_dataloader)
                print("train_epoch:{} | train_batch:{} | train_loss:{} | eval_loss:{} | train_accu:{} | eval_accu:{}"
                      "".format(epoch, batch_index, train_loss.item(), sum_eval_loss, train_accu, sum_eval_accu))
                # train_record(model, params_s, epoch, index, train_loss, eval_loss, train_accu, eval_accu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("python train.py")
    train()