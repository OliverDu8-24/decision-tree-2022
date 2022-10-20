from dataloader import *

data = merge_data()
print(data.shape)

data = process_string_col(data)
print(data.shape)

data = clean_mostly_null_data(data)
print(data.shape)

data = process_by_feature_meaning(data)
print(data.shape)

data = fill_missing_data(data)
print(data.shape)

data = set_label(data)
print(data.shape)

# data = select_dominant_col(data)
# print(data.shape)

data.to_csv('data/phase_12_cleaned.csv')
