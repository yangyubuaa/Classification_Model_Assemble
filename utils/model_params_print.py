"""
将嵌套字典进行输出
"""

from load import load_yaml

def model_params_print(configs):
    if isinstance(configs, dict) : #使用isinstance检测数据类型
        for x in range(len(configs)):
            temp_key = list(configs.keys())[x]
            temp_value = configs[temp_key]
            print("%s : %s" %(temp_key,temp_value))
            model_params_print(temp_value) #自我调用实现无限遍历

if __name__ == "__main__":
    a = load_yaml("/home/ubuntu1804/pytorch_sequence_classification/transformers_based_classification/electra_classification/electra_classification_config.yaml")
    model_params_print(a)