import pandas as pd

from utils.load import load_xlsx
from random import sample, choice

def unbalanced_process():
    source_train_data_path = "/Users/yangyu/PycharmProjects/infer_of_intent/dataset/source_data/source_all.xlsx"
    expand_train_data_path = "/Users/yangyu/PycharmProjects/infer_of_intent/dataset/expand_data/expand_all.xlsx"

    source_train_data, expand_train_data = load_xlsx(source_train_data_path), load_xlsx(expand_train_data_path)

    # 原始数据
    source_train_x, source_train_y = list(source_train_data['text']), list(source_train_data['intent'])
    # print(len(source_train_x), len(source_train_y)) 18735

    # 多标签转单标签处理
    for index in range(len(source_train_x) - 1, -1, -1):
        # 将多标签的数据替换为第一个标签
        if "," in source_train_y[index]:
            label_split = source_train_y[index].split(",")
            label = label_split[0][2:-1]
            source_train_y[index] = "['" + label + "']"

    # print(len(set(source_train_y))) 189类别

    source_train_x_without_, source_train_y_without_ = list(), list()
    # 将[]标签去掉
    for index in range(len(source_train_x)):
        if source_train_y[index] != "[]":
            source_train_x_without_.append(source_train_x[index])
            source_train_y_without_.append(source_train_y[index])

    # print(len(source_train_x_without_), len(set(source_train_y_without_))) 将所有[]数据去掉后的数据条数和类别数

    dataset = {key: list() for key in set(source_train_y_without_)}

    for i in range(len(source_train_x_without_)):
        dataset[source_train_y_without_[i]].append(source_train_x_without_[i])

    # print(dataset)



    # 处理扩充数据
    # 扩充数据
    expand_train_x, expand_train_y = list(expand_train_data['text']), list(expand_train_data['intent'])
    for index in range(len(expand_train_x) - 1, -1, -1):
        # 将多标签的数据替换为第一个标签
        if "," in expand_train_y[index]:
            label_split = expand_train_y[index].split(",")
            label = label_split[0][2:-1]
            expand_train_y[index] = "['" + label + "']"

    expand_train_x_without_, expand_train_y_without_ = list(), list()
    # 将[]标签去掉
    for index in range(len(expand_train_x)):
        if expand_train_y[index] != "[]":
            expand_train_x_without_.append(expand_train_x[index])
            expand_train_y_without_.append(expand_train_y[index])

    expand_dataset = {key: list() for key in set(expand_train_y_without_)}

    for i in range(len(expand_train_x_without_)):
        expand_dataset[expand_train_y_without_[i]].append(expand_train_x_without_[i])

    # for key, value in expand_dataset.items():
    #     print(len(value))

    source_values = list()
    for key, value in dataset.items():
        source_values.append(len(value))

    # 将多于1000样本的数据降采样
    for key in dataset.keys():
        if len(dataset[key]) > 1000:
            dataset[key] = sample(dataset[key], 1000)
        if len(dataset[key]) < 1000:
            try:
                if len(expand_dataset[key]) + len(dataset[key]) > 1000:
                    dataset[key].extend(sample(expand_dataset[key], 1000-len(expand_dataset[key])))

                else:
                    dataset[key].extend(expand_dataset[key])
            except:
                pass

    up_down_values = list()
    for key, value in dataset.items():
        up_down_values.append(len(value))
    print(list(zip(source_values, up_down_values)))

    for key in dataset.keys():
        if len(dataset[key]) < 1000:
            for i in range(1000-len(dataset[key])):
                dataset[key].append(choice(dataset[key]))

    for key, value in dataset.items():
        print(len(value))
    print(len(dataset))
    xs, ls = list(), list()
    for key, v in dataset.items():
        for value in dataset[key]:
            xs.append(value)
            ls.append(key)

    d = {
        "text":xs,
        "intent":ls
    }
    df = pd.DataFrame(d)
    df.to_excel("final.xlsx")


if __name__ == '__main__':
    unbalanced_process()