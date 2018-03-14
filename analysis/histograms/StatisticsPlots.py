import pandas as pd
import sklearn as sk
import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_general_statistics(df):

    means = df.mean(axis=1)
    medians = df.median(axis=1)
    std = df.std(axis=1)
    maxval = df.max(axis=1)
    minval = df.min(axis=1)
    skew = df.skew(axis=1)

    fig = plt.figure(figsize=(12,8))

    ax = fig.add_subplot(231)
    ax.hist(means, alpha=0.8, bins=50)
    ax.set_xlabel('Mean Intensity')
    ax.set_ylabel('Num. of Stars')

    ax = fig.add_subplot(232)
    ax.hist(medians, alpha=0.8, bins=50)
    ax.set_xlabel('Median Intensity')
    ax.set_ylabel('Num. of Stars')

    ax = fig.add_subplot(233)
    ax.hist(std, alpha=0.8, bins=50)
    ax.set_xlabel('Intensity Standard Deviation')
    ax.set_ylabel('Num. of Stars')

    ax = fig.add_subplot(234)
    ax.hist(maxval, alpha=0.8, bins=50)
    ax.set_xlabel('Maximum Intensity')
    ax.set_ylabel('Num. of Stars')

    ax = fig.add_subplot(235)
    ax.hist(minval, alpha=0.8, bins=50)
    ax.set_xlabel('Minimum Intensity')
    ax.set_ylabel('Num. of Stars')

    ax = fig.add_subplot(236)
    ax.hist(skew, alpha=0.8, bins=50)
    ax.set_xlabel('Intensity Skewness')
    ax.set_ylabel('Num. of Stars')


def plot_comparison_statistics(df):
    means1 = df[labels == 0].mean(axis=1)
    medians1 = df[labels == 0].median(axis=1)
    std1 = df[labels == 0].std(axis=1)
    maxval1 = df[labels == 0].max(axis=1)
    minval1 = df[labels == 0].min(axis=1)
    skew1 = df[labels == 0].skew(axis=1)

    means2 = df[labels == 1].mean(axis=1)
    medians2 = df[labels == 1].median(axis=1)
    std2 = df[labels == 1].std(axis=1)
    maxval2 = df[labels == 1].max(axis=1)
    minval2 = df[labels == 1].min(axis=1)
    skew2 = df[labels == 1].skew(axis=1)

    fig = plt.figure(figsize=(12, 8))

    ax = fig.add_subplot(231)
    ax.hist(means1, alpha=0.8, bins=50, color='b', normed=True, range=(-250, 250))
    ax.hist(means2, alpha=0.8, bins=50, color='r', normed=True, range=(-250, 250))
    ax.get_legend()
    ax.set_xlabel('Mean Intensity')
    ax.set_ylabel('Num. of Stars')

    ax = fig.add_subplot(232)
    ax.hist(medians1, alpha=0.8, bins=50, color='b', normed=True, range=(0, 1))
    ax.hist(medians2, alpha=0.8, bins=50, color='r', normed=True, range=(0, 1))
    ax.get_legend()
    ax.set_xlabel('Median Intensity')
    ax.set_ylabel('Num. of Stars')

    ax = fig.add_subplot(233)
    ax.hist(std1, alpha=0.8, bins=50, normed=True, color='b', range=(0, 4000))
    ax.hist(std2, alpha=0.8, bins=50, normed=True, color='r', range=(0, 4000))
    ax.get_legend()
    ax.set_xlabel('Intensity Standard Deviation')
    ax.set_ylabel('Num. of Stars')

    ax = fig.add_subplot(234)
    ax.hist(maxval1, alpha=0.8, bins=50, normed=True, color='b', range=(-10000, 50000))
    ax.hist(maxval2, alpha=0.8, bins=50, normed=True, color='r', range=(-10000, 50000))
    ax.get_legend()
    ax.set_xlabel('Maximum Intensity')
    ax.set_ylabel('Num. of Stars')

    ax = fig.add_subplot(235)
    ax.hist(minval1, alpha=0.8, bins=50, normed=True, color='b', range=(-50000, 10000))
    ax.hist(minval2, alpha=0.8, bins=50, normed=True, color='r', range=(-50000, 10000))
    ax.get_legend()
    ax.set_xlabel('Minimum Intensity')
    ax.set_ylabel('Num. of Stars')

    ax = fig.add_subplot(236)
    ax.hist(skew1, alpha=0.8, bins=50, normed=True, color='b', range=(-40, 60))
    ax.hist(skew2, alpha=0.8, bins=50, normed=True, color='r', range=(-40, 60))
    ax.get_legend()
    ax.set_xlabel('Intensity Skewness')
    ax.set_ylabel('Num. of Stars')




df = pd.read_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//merged//planets_labelled_final.csv')
print(df.head())
print(df.info())

df_planets = df.loc[df['LABEL'] == 1]
df_nonplanets = df.loc[df['LABEL'] == 0]

labels_planets = df_planets.LABEL
labels_nonplanets = df_nonplanets.LABEL
df_planets = df_planets.drop('LABEL',axis=1)
df_nonplanets = df_nonplanets.drop('LABEL',axis=1)

labels = df.LABEL
df = df.drop('LABEL',axis=1)

# Plot general statistics on both classes
#plot_general_statistics(df)

# Plot comparison histogram statistics
plot_comparison_statistics(df)
plt.show()










