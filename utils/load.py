# -*- utf-8 -*-
# @Time: 2021/4/20 9:43 上午
# @Author: yang yu
# @File: load.py.py
# @Software: PyCharm

import json
import yaml
import pandas as pd

def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as json_r:
        return json.load(json_r)


def load_yaml(yaml_path):
    config = open(yaml_path, "r", encoding="utf-8")
    cfg = config.read()
    return yaml.load(cfg)


def load_xlsx(xlsx_path):
    '''xlsx文件加载函数

    :param xlsx_path:
    :return:
    '''
    xlsx_df = pd.read_excel(xlsx_path)
    return xlsx_df