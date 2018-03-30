import numpy as np
import pandas as pd
import scipy as scp
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer
from keras.preprocessing.sequence import pad_sequences
from impyute import arima

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
                self.scaler = self.standard_scaler.fit(data)


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

    def inputeNaN(self):

        imputer = Imputer(strategy='mean',axis=1)

        return pd.DataFrame(imputer.fit_transform(self.data))

def main():

    data = readDataframe('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//original_lc//planets_labelled_final_original.csv')
    print(data.head())
    data = data.drop('LABEL',axis=1)

    # removed_nan_data = MissingValuesHandler(data).removeNaN()
    # removed_nan_data.to_csv(
    #     'C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//removed_nan_lc//planets_labelled_final_no_nan.csv',
    #     na_rep='nan', index=False)
    #
    # masked_nan_data = MissingValuesHandler(data).fillNaN(fillValue=np.nan)
    # masked_nan_data.to_csv(
    #      'C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//lc_masked_nan//planets_labelled_final_masked_nan.csv',
    #      na_rep='nan', index=False)

    # arima_nan_data = MissingValuesHandler(data).arimaNaN(5,1,0)
    # arima_nan_data.to_csv(
    #     'C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//lc_interpolated_nan//planets_labelled_final_interpolated_nan.csv',
    #     na_rep='nan', index=False)

    # imputed_nan_data = MissingValuesHandler(data).inputeNaN()
    # imputed_nan_data.to_csv(
    #     'C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//lc_imputed_nan//planets_labelled_final_imputed_nan.csv',
    #      na_rep='nan', index=False)

    if (data.isna):
        normalized_data_og = Normalizer().normalize(data, na_values=True)
        standardized_data_og = Standardizer().standardize(data, na_values=True)


    #removed_nan_data = removed_nan_data.iloc[1, :]
    #removed_nan_data = removed_nan_data.drop('LABEL', axis=0)
    #removed_nan_data = removed_nan_data.as_matrix().astype(np.float)



    normalized_data_og.to_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//original_lc/planets_labelled_final_original_normed.csv', na_rep='nan', index=False)
    standardized_data_og.to_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//original_lc/planets_labelled_final_original_std.csv', na_rep='nan', index=False)


    # for i in range(5):
	# 	print(inversed[i])


def readDataframe(path):
	return pd.read_csv(path, header=0)

if __name__ == "__main__":
    main()