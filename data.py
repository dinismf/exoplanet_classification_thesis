import os
import platform
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split, StratifiedKFold
from preprocessing import Normalizer, Standardizer

def LoadData():

    if (platform.system() == 'Windows'):
        data = pd.read_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//original_lc//planets_labelled_final_original.csv')

    elif(platform.system() == 'Darwin'):
        data = pd.read_csv('/Users/DYN/Google Drive/Intelligent_Systems_MSc/MSc_Project/data/main/original_lc/planets_labelled_final_original.csv')

    return data

def SplitData(data, test_size=0.20, preprocess='standardize'):

    y = data.LABEL

    X = data.drop('LABEL',axis=1)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)

    if (preprocess == 'standardize'):
        scaler = Standardizer()
        X_train = scaler.standardize(X_train)
        X_test = scaler.standardize(X_test)

    elif (preprocess == 'normalize'):
        scaler = Normalizer()
        X_train = scaler.normalize(X_train)
        X_test = scaler.normalize(X_test)

    # Reshape data to 3D input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


    return (X_train, y_train, X_test, y_test)

def KFoldData(data):

    X = data.VALUES
    y = data.LABEL

    skf = StratifiedKFold(n_splits=4, random_state=None, shuffle=False)

    n_splits = skf.get_n_splits()

 #
 # if __name__ == "__main__":
 #
 #     data = LoadData()
 #     X_train, y_train, X_test, y_test = SplitData(data)
 #
 #


