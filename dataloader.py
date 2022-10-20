import pandas as pd
import scipy.stats as st
from fancyimpute import KNN

pd.set_option('display.width', 300)  # 设置字符显示宽度
pd.set_option('display.max_rows', None)  # 设置显示最大行
pd.set_option('display.max_columns', None)  # 设置显示最大列，None为显示所有列


def merge_data():
    missing_values = [' ', '']
    data = pd.read_csv('data/womac_data.csv', na_values=missing_values)
    data.drop('filter_$', axis=1, inplace=True)
    data_complete = data.dropna(axis=0, how='any', inplace=False)
    data_complete.to_csv('data/womac_clean.csv')

    left = pd.read_csv('data/phase_12.csv', index_col=None)
    left.drop(['q1a', 'q1b'], axis=1, inplace=True)
    right = pd.read_csv('data/womac_clean.csv', index_col=None)
    right.drop(['q1a', 'q1b'], axis=1, inplace=True)

    data_merged = pd.merge(left, right, on='ID')

    slope = []

    for index, row in data_merged.iterrows():
        ret = st.linregress([1, 2, 3, 4], [row['WOMAC_00'], row['WOMAC_01'], row['WOMAC_02'], row['WOMAC_03']])
        slope.append(float(ret[0]))

    data_merged.insert(data_merged.shape[1], 'slope', pd.Series(slope))
    data_merged.to_csv('data/phase_12_merged.csv')
    data_origin = pd.read_csv('data/phase_12_merged.csv', index_col=None, na_values=['#NULL!', ' ', ''])
    return data_origin


def process_string_col(data):
    data_type = data.dtypes
    drop_string_index = data_type[data_type == object].index.values
    # print(drop_string_index)
    data.drop(drop_string_index, axis=1, inplace=True)
    # print(data.shape)
    return data


def clean_mostly_null_data(data, col_threshold=1300, row_threshold=200):
    data = data.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=False)
    data_clean_column = data.dropna(axis=1, thresh=col_threshold, inplace=False)
    data_clean_row = data_clean_column.dropna(axis=0, thresh=row_threshold, inplace=False)

    return data_clean_row


def process_by_feature_meaning(data_drop_row):
    # 按特征含义处理剩余缺省值
    # TODO 确认每一列的具体含义，制定不同的处理方案

    # print(f"now data shape: {data_drop_row.shape}")

    # delete useless columns (unrelated, repeated)
    useless_col_index = data_drop_row\
        .filter(regex='^(q\\w+|fs1q3\\w|fs2ct\\w+)')\
        .columns
    data_drop_useless = data_drop_row.drop(useless_col_index, axis=1, inplace=False)

    return data_drop_useless


def fill_missing_data(data):
    data_np = KNN(k=3).fit_transform(data)
    data_pd = pd.DataFrame(data_np, columns=data.columns)
    data_pd = data_pd.round(1)

    # print(dataset.isna().sum())
    # print(data_pd.isna().sum())

    return data_pd


def set_label(data, threshold=2.7):
    data['label'] = data['slope'].apply(lambda x: 1 if x > threshold else 0)

    return data


def select_dominant_col(data, corr_boundary=0.08):
    data_corr = data.corr('kendall')['label']
    data_corr = data_corr.abs().sort_values(ascending=False)

    data_corr.to_csv('corr.csv')

    not_dominant_index = data_corr[data_corr < corr_boundary].index
    # print(not_dominant_index)

    data_select_dom = data.drop(not_dominant_index, axis=1, inplace=False)

    return data_select_dom


def get_train_data(data):
    feature = data.drop(['WOMAC_02', 'WOMAC_03', 'slope', 'label'], axis=1, inplace=False)
    label = data['label'].values
    # label_name = ['front', 'middle', 'end']
    feature_name = feature.columns
    print(feature_name)

    return feature, label, feature_name
