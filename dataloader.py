import pandas as pd
from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import train_test_split
import graphviz 
import copy

data_origin = pd.read_excel('data/第1阶段.xlsx', index_col=None)

def read_data():
    print(data_origin.shape)
    return data_origin 

def womac_fillna(data):
    data['WOMAC'].fillna(data['WOMAC'].mean(), inplace=True)
    return data

def process_string_clo(data):
    data_type = data.dtypes
    drop_string_index = data_type[data_type==object].index.values
    print(drop_string_index)
    data.drop(drop_string_index, axis=1, inplace=True)
    print(data.shape)
    return data


def womac_classification(data):
    data_copy = copy.deepcopy(data)
    womac = copy.deepcopy(data_copy['WOMAC'].values)
    count = len(womac)

    womac.sort()
    range1 = womac[int(count/4)]
    range2 = womac[int(count*3/4)]

    print(data_copy.shape)

    print(f"womac splite range: {range1} {range2}")
    def get_cls(x):
        if x < range1:
            return 1
        elif x > range2:
            return 3
        else:
            return 2

    data_copy['WOMAC'] =  data_copy['WOMAC'].apply(get_cls)

    print(data_copy.shape)
    return data_copy

def have_null(data):
    df = data.isnull()
    for u in df.columns:
        if df[u].dtype==bool:
            df[u]=df[u].astype('int')
    return {
        "col_sum": df.sum(axis=0), 
        "row_sum": df.sum(axis=1),
        "index": df.columns}


# step1: fillnan  
def delete_nan_col_and_row(data_copy, col_nan_num = 210, row_nan_num = 30):
    # 列级初步处理
    col_analise, row_analise, all_col_index = have_null(data_copy).values()
    # print(col_analise)
    drop_col_index = col_analise[col_analise>col_nan_num].index
    print(f"null more than {col_nan_num}'s cols will be delete")
    print(f"drop col count: {len(drop_col_index)}")
    print(f"drop col name: {drop_col_index.values}")

    data_drop_col = data_copy.drop(drop_col_index, axis=1, inplace=False)

    # 行级初步处理
    col_analise, row_analise, all_col_index = have_null(data_drop_col).values()
    # print(row_analise)
    drop_row_index = row_analise[row_analise>row_nan_num].index
    print(f"null more than {row_nan_num}'s rows will be delete")
    print(f"drop row count: {len(drop_row_index)}")
    print(f"drop row number: {drop_row_index.values}")


    data_drop_row = data_drop_col.drop(drop_row_index, axis=0, inplace=False)

    print(f"now data shape: {data_drop_row.shape}")
    print(f"now col name: {data_drop_row.columns.values}")

    return data_drop_row

def precess_by_feature_meaning(data_drop_row):
    # 按特征含义处理剩余缺省值
    # TODO 确认每一列的具体含义，制定不同的处理方案
    data_drop_fillna = data_drop_row.dropna()
    print(f"now data shape: {data_drop_fillna.shape}")
    return data_drop_fillna




def get_train_data(data_drop_fillna):
    feature = data_drop_fillna.drop('WOMAC', axis=1, inplace=False)
    label = data_drop_fillna['WOMAC'].values
    label_name = ['front', 'middle', 'end']
    feature_name = feature.columns
    print(feature_name)

    return feature, label, feature_name