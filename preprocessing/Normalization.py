import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def main():

	data = readDataframe('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//merged//planets_labelled.csv')
	print(data.head())

	labels = data.LABEL
	data = data.drop('LABEL', axis=1)


	# prepare data for normalization
	values = data.values
	#values = values.reshape((len(values), 1))

	# train the normalization
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaler = scaler.fit(values)

	print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
	# normalize the dataset and print the first 5 rows
	normalized = scaler.transform(values)
	for i in range(5):
		print(normalized[i])


	# inverse transform and print the first 5 rows
	inversed = scaler.inverse_transform(normalized)
	for i in range(5):
		print(inversed[i])



def readDataframe(path):
	return pd.read_csv(path, header=0)


def normalizeDataframe(data):
	return data

def inverseTransformDataframe(data):
	return data



if __name__ == "__main__":
	main()





