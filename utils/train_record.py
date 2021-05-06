# -*- utf-8 -*-
# @Time: 2021/4/20 5:58 下午
# @Author: yang yu
# @File: train_record.py.py
# @Software: PyCharm
import datetime
import torch
import json

class TrainProcessRecord:
    """模型参数以及训练过程数据存储类
    """
    def __init__(self, config: dict):
        """实例化类

        config: 需要的参数
        """
        # 设置模型存储的batch阈值，每到threshold % batch == 0时，保存模型，通常该值设置为self.train_params_save_threshold的倍数
        self.train_model_save_threshold = config["train_record_settings"]["train_model_save_threshold"]

        # 设置训练过程中训练参数存储的阈值，当thredhold % batch == 0时，存储loss到self.train_records
        self.train_params_save_threshold = config["train_record_settings"]["train_params_save_threshold"]

        # 存储训练过程中得到的数据
        self.train_params_records = dict()
        # 加入模型参数
        self.train_params_records["model_params"] = config
        # 加入训练记录参数
        self.train_params_records["train_record"] = dict()

        # 设置模型和参数存储路径
        self.abs_path = config["train_record_settings"]["train_process_record_path"]

        # 设置模型名称，加入到参数存储和模型存储文件名中
        self.model_name = config["model_name"]

    def __call__(self, model, epoch, batch, *args):
        """
        在计算eval loss时调用，满足条件则存储模型参数和模型，通常调用时即存储，也就是batch % self.train_params_save_threshold == 0
        时调用此函数，一定需要传入的参数为model、 epoch和batch，其他根据任务进行存储
        """
        if batch % self.train_model_save_threshold == 0:
            # 设置模型自动存储格式
            save_path = "[{}]_epoch[{}]_batch[{}].bin".format(self.model_name, str(epoch), str(batch))
            print(save_path + " model saved! ")
            self.save_model(model, save_path)

        if batch % self.train_params_save_threshold == 0:
            # 将模型参数存储在内置数据结构中
            time_stamp = datetime.datetime.now()
            t = "time_stamp  " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S')
            self.train_params_records["train_record"][t] = (epoch, batch, *args)
            # for a in args:
            #     print(type(a))
            self.save_params()
            print("params updated!")

    def save_model(self, model, save_path="save_model.bin"):
        """
        存储模型文件，在训练结束过程中手动调用保存，存储模型时会更新record_params
        """
        # 存储模型
        torch.save(model, self.abs_path + save_path)
    
    def save_params(self):
        # 存储训练参数
        with open(self.abs_path + "save_params.json", "w") as json_w:
            json.dump(self.train_params_records, json_w, indent=2)
        