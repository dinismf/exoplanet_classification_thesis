import pandas as pd
import scipy as scp
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from keras.preprocessing.sequence import pad_sequences

class Normalizer:

	scaler = None

	def __init__(self):
		self.scaler = MinMaxScaler(feature_range=(0,1))

	def normalize(self, data):
		self.scaler = self.scaler.fit(data.values)

		return self.scaler.transform(data.values)

	def inverseNormalization(self, normed_data):
		return self.scaler.inverse_transform(normed_data.values)

class Standardizer:
    standard_scaler = None

    def __init__(self):
        self.standard_scaler = StandardScaler()

    def standardize(self, data):
        self.scaler = self.standard_scaler.fit(data.values)
        return self.standard_scaler.transform(data.values)

    def inverseStandardization(self, standard_data):
        return self.standard_scaler.inverse_transform(standard_data.values)


class MissingValuesHandler():

    data = None
    def __init__(self, data):
        self.data = data

    def fillNaN(self, fillValue):
        return self.data.fillna(fillValue)

    def removeNaN(self):
        return self.data.dropna(axis=1)

    def interpolateNaN(self):
        return self.data.interpolate(method='barycentric')

def main():

    data = readDataframe('/Users/DYN/Google Drive/Intelligent_Systems_MSc/MSc_Project/data/main/original_lc/planets_labelled_final_original.csv')
    print(data.head())

    removed_nan_data = MissingValuesHandler(data).removeNaN()
    removed_nan_data.to_csv(
        '/Users/DYN/Google Drive/Intelligent_Systems_MSc/MSc_Project/data/main/removed_nan_lc/planets_labelled_final_no_nan.csv',
        na_rep='nan', index=False)

    # normalized_data = Normalizer().normalize(data)
    # for i in range(5):
    #     print(normalized_data[i])



    # for i in range(5):
	# 	print(inversed[i])


def readDataframe(path):
	return pd.read_csv(path, header=0)

if __name__ == "__main__":
    main()