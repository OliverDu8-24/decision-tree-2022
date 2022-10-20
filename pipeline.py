from dataloader import *


def pipeline():
    data = merge_data()
    data = process_string_col(data)
    data = clean_mostly_null_data(data)
    data = process_by_feature_meaning(data)
    data = fill_missing_data(data)
    data = set_label(data)
    data = select_dominant_col(data)

    return get_train_data(data)
