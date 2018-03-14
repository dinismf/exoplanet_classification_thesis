import pandas as pd
import scipy as scp

def FillNA(data, val):
    return data.fillna(val)

def RemoveNA(data):
    return data.dropna(axis=1)

def InterpolateNA(data):
    return data.interpolate(method='barycentric')


data = pd.read_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//merged//planets_labelled.csv', header=0)
data.set_index("LABEL")
data = data.loc[data['LABEL'] == 1]

data_filled_na = FillNA(data, -1)
data_removed_na = RemoveNA(data)
data_interpolated_barycentric = InterpolateNA(data)




print(data.head())










