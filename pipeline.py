from dataloader import *


def preprocess():
    data = merge_data()
    data = process_string_col(data)
    data = clean_mostly_null_data(data)
    data = process_by_feature_meaning(data)
    data = fill_missing_data(data)
    return data


def generate(slope=3.0, corr_boundary=0.07):
    data = pd.read_csv('data/phase_12_filled.csv', index_col=None, na_values=['#NULL!', ' ', ''])
    data = set_label(data, slope)
    data = select_essential_col(data)
    # data = select_dominant_col(data, corr_boundary)
    data = get_train_data(data)
    return data
