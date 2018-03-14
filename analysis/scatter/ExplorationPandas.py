import pandas as pd
import sklearn as sk
import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline

df = pd.read_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//planets_labelled.csv')

print(df.head())
print(df.info())



labels = df.LABEL
df = df.drop('LABEL',axis=1)


n_rows = len(df[:])
n_sample_rows = 40;

x = np.array(range(len(df.columns)))

random_sample_df = df.sample(n=n_sample_rows)

y = random_sample_df.iloc[0,:]


plot_cols = 2;
plot_rows = int(n_sample_rows/plot_cols)
fig, axes = plt.subplots(nrows=plot_rows, ncols=plot_cols, sharex=True)
#fig, axes = plt.subplots(nrows=int(n_sample_rows), sharex=True)

for i in range(plot_rows):

    y = random_sample_df.iloc[i, :]

    axes[i,0].scatter(x=x,y=y,s=1)
    axes[i,1].plot(x,y)

#plt.scatter(x=x, y=y, s=1)
#plt.plot(x, y)

# Output to figure png
#plt.savefig('line_scatter_plots.png', bbox_inches='tight')

# Display
plt.show()












