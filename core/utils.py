from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import numpy as np


def get_onehot_encoder(sparse=False, handle_unknown="ignore"):
    return OneHotEncoder(sparse=sparse, handle_unknown=handle_unknown)


def get_oridinal_encoder(unknown_value=np.nan, handle_unknown="use_encoded_value"):
    return OrdinalEncoder(unknown_value=unknown_value, handle_unknown=handle_unknown)
