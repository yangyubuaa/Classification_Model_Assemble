### 文本分类问题模型实现
---
.
├── config # 设置文件放置，弃用
│   ├── __init__.py
│   └── __pycache__
│       └── __init__.cpython-38.pyc
├── data_preprocess # 数据预处理文件放置
│   ├── Cross_validation # 交叉验证实现
│   │   ├── cross_validation.py
│   │   ├── __init__.py
│   │   └── __pycache__
│   │       ├── cross_validation.cpython-38.pyc
│   │       └── __init__.cpython-38.pyc
│   ├── Dataset # 序列数据集类实现
│   │   ├── dataset.py
│   │   ├── __init__.py
│   │   └── __pycache__
│   │       ├── dataset.cpython-38.pyc
│   │       └── __init__.cpython-38.pyc
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-38.pyc
│   ├── TensorSort # tensor排序工具
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   └── tensor_sort.cpython-38.pyc
│   │   └── tensor_sort.py
│   └── Tokenizer # 各种tokenizer实现
│       ├── __init__.py
│       ├── label_tokenizer.py
│       ├── __pycache__
│       │   ├── __init__.cpython-38.pyc
│       │   ├── label_tokenizer.cpython-38.pyc
│       │   └── tokenizer.cpython-38.pyc
│       └── tokenizer.py
├── dataset # 根据不同任务在此预处理数据
│   ├── eval_data
│   │   └── eval.xlsx
│   ├── expand_data
│   │   └── expand_all.xlsx
│   ├── final_.xlsx
│   ├── final.xlsx
│   ├── label2index.json
│   ├── preprocess_config.yaml
│   ├── preprocess.py
│   ├── __pycache__
│   │   └── preprocess.cpython-38.pyc
│   ├── source_data
│   │   └── source_all.xlsx
│   ├── unbalanced.py
│   └── vocab2index.json
├── readme.md # 说明文档
├── simple_classification # 未使用预训练模型的简单分类器
│   ├── __init__.py
│   ├── lstm_base # 直接最长序列填充的lstm模型
│   │   ├── __init__.py
│   │   ├── lstm_base_config.yaml
│   │   ├── model.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   └── model.cpython-38.pyc
│   │   └── train.py
│   ├── lstm_pack # 进行pack后的lstm模型
│   │   ├── __init__.py
│   │   ├── lstm_pack_config.yaml
│   │   ├── model.py
│   │   ├── predict.py
│   │   ├── predict_result.xlsx
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   └── model.cpython-38.pyc
│   │   └── train.py
│   └── __pycache__
│       └── __init__.cpython-38.pyc
├── test.py # 已弃用
├── transformers_based_classification # 基于transformer架构的预训练模型分类
│   ├── bert_classification # 基于bert
│   │   ├── arg.py
│   │   ├── bert_classification_config.yaml
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   └── model.cpython-38.pyc
│   │   └── train.py
│   ├── electra_classification # 基于electra
│   │   ├── electra_classification_config.yaml
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── train_recordsave_params.json
│   ├── __init__.py
│   └── __pycache__
│       └── __init__.cpython-38.pyc
├── tree.txt # 目录树文件
└── utils # 工具类
    ├── __init__.py
    ├── load.py # 各种类型文件加载类
    ├── model_params_print.py # 模型参数打印类
    ├── __pycache__
    │   ├── __init__.cpython-38.pyc
    │   ├── load.cpython-38.pyc
    │   └── train_record.cpython-38.pyc
    ├── train_record.py # 训练过程记录类
    └── train_record_visualize.py # 训练过程数据可视化类

30 directories, 74 files
