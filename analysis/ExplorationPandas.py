import pandas as pd
import sklearn as sk
import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline



df = pd.read_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//output.csv')
df = df.transpose()

print(df.head())
print(df.info())


# Plot the light curves of all 37 exoplanet stars
exoplanet_fig = plt.figure(figsize=(12,40))

x = np.array(range(len(df.columns)))

#plt.scatter(x, df.iloc[0, :], s=1)
plt.plot(x, df.iloc[0, :])


plt.show()












