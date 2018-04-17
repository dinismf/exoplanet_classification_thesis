import pandas as pd
import sklearn as sk
import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline

#df = pd.read_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//original_lc//planets_labelled_final_original.csv')
df = pd.read_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//kepler_planets//confirmed//candidates.csv')
#df = pd.read_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//lc_kalman_nan//kalman_nan_data.csv')
#df = pd.read_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//lc_movingaverage_nan//movingaverage_nan_data.csv')
#df = pd.read_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//lc_interpolated_nan//planets_labelled_final_interpolated_nan.csv')

print(df.head())
print(df.info())

df_planets = df.loc[df['LABEL'] == 1]
df_nonplanets = df.loc[df['LABEL'] == 0]

labels_planets = df_planets.LABEL
labels_nonplanets = df_nonplanets.LABEL
df_planets = df_planets.drop('LABEL',axis=1)
df_nonplanets = df_nonplanets.drop('LABEL',axis=1)


# Number of rows to sample
n_sample_rows = 10;

# Timeseries max length
x_planets = np.array(range(len(df_planets.columns)))
x_nonplanets = np.array(range(len(df_nonplanets.columns)))

# Randomly sample n rows from each dataframe
random_sample_df_planets = df_planets.sample(n=n_sample_rows)
random_sample_df_nonplanets = df_nonplanets.sample(n=n_sample_rows)


plot_cols = 2;
plot_rows = int(n_sample_rows)

fig, axes = plt.subplots(figsize=(20,40), nrows=plot_rows, ncols=plot_cols)
#fig, axes = plt.subplots(nrows=int(n_sample_rows), sharex=True)

for i in range(plot_rows):

    y = df_planets.iloc[i,:]
    y2 = df_nonplanets.iloc[i,:]

    axes[i,0].scatter(x=x_planets,y=y,s=1)

    axes[i,1].scatter(x=x_nonplanets,y=y2, s=1)

#plt.scatter(x=x, y=y, s=1)
#plt.plot(x, y)

# Output to figure png
#plt.savefig('line_scatter_plots2.png', bbox_inches='tight')

# Display
plt.show()












