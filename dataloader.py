import pandas as pd
import copy

pd.set_option('display.width', 300)  # 设置字符显示宽度
pd.set_option('display.max_rows', None)  # 设置显示最大行
pd.set_option('display.max_columns', None)  # 设置显示最大列，None为显示所有列

data_origin = pd.read_excel('data/第1阶段.xlsx', index_col=None)


def read_data():
    # print(data_origin.shape)
    return data_origin


def womac_fillna(data):
    data['WOMAC'].fillna(data['WOMAC'].mean(), inplace=True)
    return data


def process_string_clo(data):
    data_type = data.dtypes
    drop_string_index = data_type[data_type == object].index.values
    # print(drop_string_index)
    data.drop(drop_string_index, axis=1, inplace=True)
    # print(data.shape)
    return data


def womac_classification(data):
    data_copy = copy.deepcopy(data)
    womac = copy.deepcopy(data_copy['WOMAC'].values)
    count = len(womac)

    womac.sort()
    range1 = womac[int(count / 4)]
    range2 = womac[int(count * 3 / 4)]

    # print(data_copy.shape)

    # print(f"womac split range: {range1} {range2}")

    def get_cls(x):
        if x < range1:
            return 1
        elif x > range2:
            return 3
        else:
            return 2

    data_copy['WOMAC'] = data_copy['WOMAC'].apply(get_cls)

    # print(data_copy.shape)
    return data_copy


def have_null(data):
    df = data.isnull()
    for u in df.columns:
        if df[u].dtype == bool:
            df[u] = df[u].astype('int')
    return {
        "col_sum": df.sum(axis=0),
        "row_sum": df.sum(axis=1),
        "index": df.columns}


# step1: fillnan  
def delete_nan_col_and_row(data_copy, col_nan_num=210, row_nan_num=30):
    # 列级初步处理
    col_analise, row_analise, all_col_index = have_null(data_copy).values()
    # print(col_analise)
    drop_col_index = col_analise[col_analise > col_nan_num].index
    # print(f"null more than {col_nan_num}'s cols will be delete")
    # print(f"drop col count: {len(drop_col_index)}")
    # print(f"drop col name: {drop_col_index.values}")

    data_drop_col = data_copy.drop(drop_col_index, axis=1, inplace=False)

    # 行级初步处理
    col_analise, row_analise, all_col_index = have_null(data_drop_col).values()
    # print(row_analise)
    drop_row_index = row_analise[row_analise > row_nan_num].index
    # print(f"null more than {row_nan_num}'s rows will be deleted")
    # print(f"drop row count: {len(drop_row_index)}")
    # print(f"drop row number: {drop_row_index.values}")

    data_drop_row = data_drop_col.drop(drop_row_index, axis=0, inplace=False)

    # print(f"now data shape: {data_drop_row.shape}")
    # print(f"now col name: {data_drop_row.columns.values}")

    return data_drop_row


def process_by_feature_meaning(data_drop_row):
    # 按特征含义处理剩余缺省值
    # TODO 确认每一列的具体含义，制定不同的处理方案

    # print(f"now data shape: {data_drop_row.shape}")

    # delete useless columns (unrelated, repeated)
    useless_col_index = data_drop_row.filter(regex='^(q1b|center|fs1q3\\w|fs1q4|fs1w\\w+)').columns
    data_drop_useless = data_drop_row.drop(useless_col_index, axis=1, inplace=False)

    # print(f"now data shape: {data_drop_useless.shape}")

    # fill with average value
    fill_avg_index = data_drop_useless.filter(regex='^(age|fs1c2y|fs1ct1|fs1ct2|fs1ct6z1|fs1ct7y2)').columns
    data_fill_avg = data_drop_useless
    for index in fill_avg_index:
        data_fill_avg[index].fillna(data_fill_avg[index].mean(), inplace=True)

    # fill with mode value
    data_fill_mode = data_fill_avg.fillna(data_fill_avg.mode().iloc[0], inplace=False)

    print(f"now data shape: {data_fill_mode.shape}")

    return data_fill_mode

    # data_drop_fillna = data_drop_row.dropna()
    # print(f"now data shape: {data_drop_fillna.shape}")
    # return data_drop_fillna

    # col_na_num = have_null(data_fill_mode).get("col_sum")
    # col_na_num.to_csv('blank_column.csv')


def select_dominant_col(data_fill_mode):
    data_corr = data_fill_mode.corr('kendall')['WOMAC']
    # print(data_corr)

    corr_boundary = 0.35

    not_dominant_index = data_corr[(-corr_boundary < data_corr) & (data_corr < corr_boundary)].index
    # print(not_dominant_index)

    data_select_dom = data_fill_mode.drop(not_dominant_index, axis=1, inplace=False)

    data_select_dom.to_csv('phase_1_reduced.csv')

    return data_select_dom


def get_train_data(data_fill_mode):
    feature = data_fill_mode.drop('WOMAC', axis=1, inplace=False)
    label = data_fill_mode['WOMAC'].values
    label_name = ['front', 'middle', 'end']
    feature_name = feature.columns
    print(feature_name)

    return feature, label, feature_name
