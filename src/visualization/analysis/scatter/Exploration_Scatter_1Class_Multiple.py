import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%matplotlib inline

#df = pd.read_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//original_lc//planets_labelled_final_original.csv')
#df = pd.read_csv('C:\\Users\\DYN\\Desktop\\exoplanet_classification_repo\\data\\final_new_fps.csv')
df = pd.read_csv('C:\\Users\\DYN\\Desktop\\exoplanet_classification_repo\\data\\binned_confirmed.csv')

#df = pd.read_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//lc_kalman_nan//kalman_nan_data.csv')
#df = pd.read_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//lc_movingaverage_nan//movingaverage_nan_data.csv')
#df = pd.read_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//lc_interpolated_nan//planets_labelled_final_interpolated_nan.csv')

#X_train, y_train, pvals = data.LoadPickledData()

#X_train = pd.DataFrame(X_train)
#y_train = pd.DataFrame(y_train, columns=['LABEL'])
#df = y_train.join(X_train)

print(df.head())
#print(df.info())

df = df.loc[df['LABEL'] == 1]

labels_planets = df.LABEL
df = df.drop('LABEL',axis=1)

# Number of rows to sample
n_sample_rows = 5;

# Timeseries max length
x_planets = np.array(range(len(df.columns)))

# Randomly sample n rows from each dataframe
random_sample_df_planets = df.sample(n=n_sample_rows, random_state=np.random.RandomState())

plot_cols = 1;
plot_rows = int(n_sample_rows)

fig, axes = plt.subplots(figsize=(20,40), nrows=plot_rows, ncols=plot_cols)
#fig, axes = plt.subplots(nrows=int(n_sample_rows), sharex=True)

for i in range(plot_rows):

    y = random_sample_df_planets.iloc[i,:]

    axes[i].scatter(x=x_planets,y=y,s=1)

#plt.scatter(x=x, y=y, s=1)
#plt.plot(x, y)

# Output to figure png
#plt.savefig('line_scatter_plots2.png', bbox_inches='tight')

# Display
plt.show()












