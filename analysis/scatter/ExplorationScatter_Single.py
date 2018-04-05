import pandas as pd
import sklearn as sk
import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline

#df = pd.read_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//original_lc//planets_labelled_final_original.csv')
df = pd.read_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//original_lc//planets_labelled_final_smoothed_std.csv')
#df = pd.read_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//lc_movingaverage_nan//movingaverage_nan_data.csv')

print(df.head())
print(df.info())

x = np.array(range(len(df.columns)))

plt.scatter(x, df.iloc[30, :], s=1)
#plt.plot(x, df.iloc[0, :])

plt.show()












