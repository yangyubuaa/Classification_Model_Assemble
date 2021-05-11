import pandas as pd
import sys
sys.path.append("../")
from utils.load import load_xlsx, load_yaml
from random import sample, choice
import pandas as pd

class DataClear:
    def __init__(self, configs):
        self.configs = configs
        self.use_expand = self.configs["use_expand"]
        if self.use_expand:
            self.dirty_path = self.configs["expand_dirty_path"]
        else:
            self.dirty_path = self.configs["source_dirty_path"]


    def clear(self):
        # 读取数据
        dirty_df = load_xlsx(self.dirty_path)
        dirty_labels = list()
        true_labels = list()
        labels = list(set(list(dirty_df["intent"])))
        print(labels)
        print(len(labels))
        for label in labels:
            if "_" in label:
                true_labels.append(label)
        # print(true_labels)

        print(len(true_labels))


if __name__ == '__main__':
    configs = load_yaml("clear_config.yaml")

    data_clear = DataClear(configs)
    data_clear.clear()