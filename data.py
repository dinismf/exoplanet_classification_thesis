import os
import platform
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split, StratifiedKFold
from preprocessing import *

def LoadOriginalData():
    """

    Returns: Dataframe X of all timeseries samples, and Series of labels y (1 = Exoplanet, 0 = No Exoplanet)

    """

    if (platform.system() == 'Windows'):
        data = pd.read_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//original_lc//planets_labelled_final_original.csv')
        #data = pd.read_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//main//original_lc//planets_labelled_final_original_equal.csv')

    elif(platform.system() == 'Darwin'):
        data = pd.read_csv('/Users/DYN/Google Drive/Intelligent_Systems_MSc/MSc_Project/data/main/original_lc/planets_labelled_final_original.csv')

    y = data.LABEL
    X = data.drop('LABEL',axis=1)

    return X, y

def SplitData(X, y, test_size=0.20, val_set = False):
    """

    Args:
        X:
        y:
        test_size:
        val_set:

    Returns:

    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42, stratify=y)

    if val_set:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state= 42, stratify=y_train)


    X_train =  X_train.as_matrix().astype(np.float)

    if val_set:
        X_val = X_val.as_matrix().astype(np.float)

    X_test =  X_test.as_matrix().astype(np.float)

    if val_set:
        return (X_train, y_train, X_val, y_val, X_test, y_test)
    else:
        return (X_train, y_train, X_test, y_test)

def KFoldData(data):
    """

    Args:
        data:
    """

    X = data.VALUES
    y = data.LABEL

    skf = StratifiedKFold(n_splits=4, random_state=None, shuffle=False)

    n_splits = skf.get_n_splits()


def GenerateDataset(og_X, og_y, filename, filename_words):

    root = 'data/'
    cadence = filename_words[0]
    scaling = filename_words[1]

    try:
        nan_handling = filename_words[2]
    except:
        nan_handling = None

    try:
        oversampling = filename_words[3]
    except:
        oversampling = None



    if (os.path.isfile(root + filename + '.csv')):
        print('Dataset: ' + filename + ' already exists')
    else:

        X_processed = og_X
        y_processed = og_y

        # Standardize of Normalise the data
        if (scaling == 'std'):
            X_processed = Standardizer().standardize(og_X, na_values=True)

        elif (scaling == 'norm'):

            X_processed = Normalizer().normalize(og_X, na_values=True)

        # Handle the missing values in the timeseries data
        handler = MissingValuesHandler(X_processed)

        if (nan_handling == 'nanmasked'):
            X_processed = handler.fillNaN(fillValue=0)
        elif (nan_handling == 'nanremoved'):
            X_processed = handler.removeNaN()

        elif (nan_handling == 'nanimputed'):
            X_processed = handler.imputeNaN()

        elif (nan_handling == 'naninterpolated'):
            nan_handling = None
        elif (nan_handling == 'nanarima'):
            nan_handling = None

        # Oversample
        if (oversampling == 'SMOTE' and nan_handling is not None):
            X_processed, y_processed = Oversampler().OversampleSMOTE(X_processed, og_y)
        elif (oversampling == 'ADASYN' and nan_handling is not None):
            X_processed, y_processed = Oversampler().OversampleADASYN(X_processed, og_y)

        # Merge new dataframe
        y_processed = pd.DataFrame(y_processed, columns=['LABEL'])
        new_df = y_processed.join(X_processed)

        # Output the processed dataframe to .csv
        path = str (root + filename + '.csv')
        new_df.to_csv(path, na_rep='nan', index=False)
        print('Dataset: ' + filename + ' created successfully')





if __name__ == "__main__":

    X, y = LoadOriginalData()

        # Read filename and store lines in list
    filenames = list(open('data/datasets.txt'))

    for fn in filenames:
        words = fn.rstrip('\n').split("_")

        GenerateDataset(X, y, fn.rstrip('\n'), words)




        # Split data
        # X_train, y_train, X_test, y_test = SplitData(X,y, test_size=0.2)
        # X_train, y_train, X_val, y_val, X_test, y_test = SplitData(X, y, test_size=0.2, val_set=True)




