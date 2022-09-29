from dataloader import *


def pipeline():
    data = read_data()
    data = womac_fillna(data)
    data = process_string_clo(data)
    data = womac_classification(data)
    data = delete_nan_col_and_row(data)
    data = process_by_feature_meaning(data)
    data = select_dominant_col(data)

    return get_train_data(data)
