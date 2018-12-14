import numpy as np
import pandas as pd
import scipy as scp
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer
from keras.preprocessing.sequence import pad_sequences
from impyute import arima

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

class Normalizer:

	scaler = None

	def __init__(self):
		self.scaler = MinMaxScaler(feature_range=(0,1))

	def normalize(self, data, na_values = False):
            if (na_values):
                normalized_df = (data - data.min()) / (data.max() - data.min())
                return normalized_df

            else:
                self.scaler = self.scaler.fit(data)
                return self.scaler.transform(data)


	def inverseNormalization(self, normed_data):
		return self.scaler.inverse_transform(normed_data)

class Standardizer:
    standard_scaler = None

    def __init__(self):
        self.standard_scaler = StandardScaler()

    def standardize(self, data, na_values):

            if (na_values):
                standardized_df = (data - data.mean()) / data.std()
                return standardized_df

            else:
                self.standard_scaler = self.standard_scaler.fit(data)


            return self.standard_scaler.transform(data)

    def inverseStandardization(self, standard_data):
        return self.standard_scaler.inverse_transform(standard_data)


class MissingValuesHandler():

    data = None
    def __init__(self, data):
        self.data = data

    def fillNaN(self, fillValue):
        return self.data.fillna(fillValue)

    def removeNaN(self):
        return self.data.dropna(axis=1)

    def interpolateNaN(self):
        return self.data.interpolate()

    def arimaNaN(self, p, d, q):
        return arima(self.data, p, d, q)

    def imputeNaN(self):

        imputer = Imputer(strategy='mean',axis=1)

        return pd.DataFrame(imputer.fit_transform(self.data))

class Smoother():
     def __init__(self, df):
         self.data = df

     def MovingAverage(self):
         data_averaged = self.data.rolling(window=4000, center=True)
         data_rollingmean = data_averaged.mean()
         return data_rollingmean


class Oversampler():

    def __init__(self):
        pass

    def OversampleNaiveRandom(self, X, y):
        ros = RandomOverSampler(random_state=0)

        X_resampled, y_resampled = ros.fit_sample(X, y)

        return X_resampled, y_resampled


    def OversampleSMOTE(self, X, y, kind=None):
        if (kind is not None):
            X_resampled, y_resampled = SMOTE(kind=kind).fit_sample(X, y)
        else:
            X_resampled, y_resampled =  SMOTE().fit_sample(X, y)

        return pd.DataFrame(X_resampled), pd.Series(y_resampled)

    def OversampleADASYN(self, X, y):

        X_resampled, y_resampled = ADASYN().fit_sample(X, y)
        return pd.DataFrame(X_resampled), pd.Series(y_resampled)
