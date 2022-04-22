import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv_data(data_path, label_column, random_state=1):

    data = pd.read_csv(data_path)
    train, test = train_test_split(data, random_state=random_state)
    train_x = train.drop([label_column], axis=1)
    test_x = test.drop([label_column], axis=1)
    train_y = train[[label_column]]
    test_y = test[[label_column]]
    return train_x, train_y, test_x, test_y
