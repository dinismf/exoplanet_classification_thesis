import pandas as pd
import sklearn as sk
import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline

def plot_timeseries(series, title):
    plt.plot(np.arange(len(series)), series)
    plt.title(title)
    plt.xlabel('Time step')
    plt.ylabel('Luminosity')
    plt.show()

# Read training data
df = pd.read_csv('/Users/DYN/Google Drive/Intelligent_Systems_MSc/MSc_Project/data/datasets/kaggle_exoplanet_timeseries/exoTrain.csv')

print(df.head())
print(df.info())

# Convert the target labels to 0 (No Exoplanet ) and 1 (Exoplanet) respectively
df['LABEL'] = df['LABEL'].replace([1], [0])
df['LABEL'] = df['LABEL'].replace([2], [1])

# Calculate exo planet positive rate
train_pr = sum(df.LABEL == 1) / float(len(df.LABEL))
print('Training set positive exo planet rate', train_pr)


plot_timeseries(df.iloc[6, 1:], 'Time series data: exoplanet confirmed')
plot_timeseries(df.iloc[100, 1:], 'Time series data: no exoplanet confirmed')
