# decision-tree-2022

## 项目目录

```
.
├── README.md
├── data
│   └── 第1阶段.xlsx
├── dataloader.py
├── pipeline.py
├── results
├── test.ipynb
├── test.py
└── train.py
```

### data

放置数据

### result

放置决策树模型结果，以及决策树形状图片

### dataloader.py

放置各类数据处理函数

### pipeline.py

模型的数据处理模块，按序列调用dataloader中的处理函数，每次处理可称为一个极端，上一阶段输出为下一阶段输入。

示例：

```python
from dataloader import *

def pipeline():
    data = read_data()
    data = womac_fillna(data)
    data = process_string_clo(data)
    data = womac_classification(data)
    data = delete_nan_col_and_row(data)
    data = precess_by_feature_meaning(data)
    return get_train_data(data)
```

* 在`dataloader.py`中新增函数，并在`pipline.py`中调用以对数据进行进一步处理

### train.py

训练模块，简要介绍：

1. 在其中调用`pipline`进行数据处理
2. 在其中定义了`train`函数作为每次实验的训练函数
3. 在其中修改超参数进行调参

### test.ipynb

快速实验