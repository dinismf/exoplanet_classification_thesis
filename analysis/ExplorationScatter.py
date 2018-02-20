import pandas as pd
import sklearn as sk
import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline


df = pd.read_csv('/Users/DYN/Google Drive/Intelligent_Systems_MSc/MSc_Project/data/datasets/kaggle_exoplanet_timeseries/exoTrain.csv',index_col=-1)

print(df.head())
print(df.info())

# Convert the target labels to 0 (No Exoplanet ) and 1 (Exoplanet) respectively
df['LABEL'] = df['LABEL'].replace([1], [0])
df['LABEL'] = df['LABEL'].replace([2], [1])

labels = df.LABEL
df = df.drop('LABEL',axis=1)

x = np.array(range(4608))

# Plot the light curves of all 37 exoplanet stars
exoplanet_fig = plt.figure(figsize=(12,40))

for i in range (37):
     ax = exoplanet_fig.add_subplot(13, 3, i + 1)
     ax.scatter(x, df[labels == 1].iloc[i, :], s=1)

# Plot the light curves for 37 non-exoplanet stars
for i in range(37):
    ax = exoplanet_fig.add_subplot(13, 3, i + 1)
    ax.scatter(x, df.iloc[i, :], s=1)

plt.show()