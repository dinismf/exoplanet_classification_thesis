import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Standardizer:
    standard_scaler = None

    def __init__(self):
        self.standard_scaler = StandardScaler()

    def standardize(self, data, na_values):

        if na_values:
            standardized_df = (data - data.mean()) / data.std()
            return standardized_df

        else:
            self.standard_scaler = self.standard_scaler.fit(data)

        return self.standard_scaler.transform(data)
