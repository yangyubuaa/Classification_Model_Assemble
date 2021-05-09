import torch
import torch.nn as nn
import argparse
import sys
import time
sys.path.append("../..")
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.functional import cross_entropy

# 混合精度训练
from torch.cuda.amp import autocast, GradScaler

from transformers import  AutoTokenizer

from utils.load import load_yaml
from utils.train_record import TrainProcessRecord
from utils.model_params_print import model_params_print
from dataset.preprocess import Preprocess
from transformers_based_classification.electra_classification.model import ElectraClassification
from data_preprocess.Dataset.dataset import BertSequenceDataset
from data_preprocess.Tokenizer.label_tokenizer import ClassificationLabelTokenizer


def train():
    # 是否使用gpu加速
    use_cuda = True if torch.cuda.is_available() else False
    # 查看可用的GPU数量
    if use_cuda:
        device_nums = torch.cuda.device_count()
        print("use {} GPUs!".format(device_nums))

    # 预处理以及模型参数读取（后面需要修改，因为需要读取两个文件，耦合性过高）
    train_configs = load_yaml("electra_classification_config.yaml")

    print("模型参数如下：")
    model_params_print(train_configs)

    params = load_yaml(train_configs["path"]["preprocess_config_path"])

    # 根据参数实例化数据集类（当前该类需要训练集和测试集才能够初始化，需要修改）
    p = Preprocess(params)

    # 得到原始数据集文本和标签
    source_train_x, source_train_y = p.get_train_data()
    source_eval_x, source_eval_y = p.get_eval_data()

    # 初始化bert输入和label的tokenizer
    electratokenizer = AutoTokenizer.from_pretrained(train_configs["path"]["electra_path"])
    labeltokenizer = ClassificationLabelTokenizer(params["tokenized_path"]["label2index_json_path"])

    # 将训练集进行tokenize
    train_x_tokenized = electratokenizer(source_train_x, padding=True, truncation=True, return_tensors="pt")
    train_y_tokenized = labeltokenizer(source_train_y)

    # 将测试集进行tokenize
    eval_x_tokenized = electratokenizer(source_eval_x, padding=True, truncation=True, return_tensors="pt")
    eval_y_tokenized = labeltokenizer(source_eval_y)

    # 创建训练集和测试集（测试集类，如果有明确划分的训练集和测试集，那么分别进行初始化，如果只有训练集，那么可以进行交叉验证）
    train_set = BertSequenceDataset(train_x_tokenized, train_y_tokenized)
    eval_set = BertSequenceDataset(eval_x_tokenized, eval_y_tokenized)

    # 读取模型参数初始化模型
    model = ElectraClassification(train_configs)

    # 如果使用cuda，那么进行多GPU训练
    if use_cuda:
        model = nn.DataParallel(model, device_ids=list(range(device_nums)))
        # 将模型放在第0个GPU
        model = model.cuda(device=0)

    # 获取训练参数
    epoch_param = train_configs["train_params"]["epoch"]
    batch_size_param = train_configs["train_params"]["batch_size"]
    learning_rate_param = train_configs["train_params"]["learning_rate"]

    # 使用dataloader加载训练集和测试集
    train_dataloader = DataLoader(train_set, batch_size=batch_size_param, shuffle=True)
    eval_dataloader = DataLoader(eval_set, batch_size=batch_size_param)

    # 创建优化器
    optmizer = Adam(model.parameters(), lr=learning_rate_param)

    # 创建损失函数
    loss_f = nn.CrossEntropyLoss()

    train_record = TrainProcessRecord(train_configs)

    scaler = GradScaler()

    train_start = time.time()
    for epoch in range(epoch_param):
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


            with autocast():
                train_y_predict = model(input_ids, token_type_ids, attention_mask)
                train_y = train_y.squeeze()
                train_loss = loss_f(train_y_predict, train_y)

            scaler.scale(train_loss).backward()

            scaler.step(optmizer)
            scaler.update()
            # 不使用混合精度训练
            # optmizer.step()

            # 每 train_params_save_threshold 个batch进行测试集预测和存储
            if batch_index % train_configs["train_record_settings"]["train_params_save_threshold"] == 0:
                model.eval()
                with torch.no_grad():
                    train_predict = torch.argmax(train_y_predict, 1)
                    train_accu = int((train_y == train_predict).sum()) / len(train_y)
                    sum_eval_accu = 0
                    sum_eval_loss = 0

                    eval_start = time.time()

                    for e_i, eval_batch in enumerate(eval_dataloader):
                        eval_input_ids, eval_token_type_ids, eval_attention_mask, eval_y = eval_batch
                        eval_y = eval_y.squeeze()
                        if use_cuda:
                            eval_input_ids, eval_token_type_ids, eval_attention_mask, eval_y = \
                                eval_input_ids.cuda(device=0), eval_token_type_ids.cuda(device=0), eval_attention_mask.cuda(device=0), eval_y.cuda(device=0)
                        # print(e_i)
                        eval_y_predict = model(eval_input_ids, eval_token_type_ids, eval_attention_mask)
                        eval_loss = cross_entropy(eval_y_predict, eval_y)

                        eval_predict = torch.argmax(eval_y_predict, 1)
                        eval_accu = int((eval_y == eval_predict).sum()) / len(eval_y)
                        sum_eval_accu = sum_eval_accu + eval_accu
                        sum_eval_loss = sum_eval_loss + eval_loss.item()
                        optmizer.zero_grad()
                        torch.cuda.empty_cache()
                    sum_eval_accu = sum_eval_accu / len(eval_dataloader)
                    sum_eval_loss = sum_eval_loss / len(eval_dataloader)

                    eval_end = time.time()

                    eval_time = eval_end - eval_start

                    print("train_epoch:{} | train_batch:{} | train_loss:{} | eval_loss:{} | train_accu:{} | eval_accu:{}"
                          "eval time:{}".format(epoch, batch_index, train_loss.item(), sum_eval_loss, train_accu, sum_eval_accu, eval_time))
                    train_record(model, epoch, batch_index, train_loss.item(), eval_loss.item(), train_accu, eval_accu, eval_time)
                model.train()
    train_end = time.time()
    train_time = train_end - train_start
    train_record.save_time(train_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("python train.py")
    train()