import os
import platform
import pandas as pd
import numpy as np
import pickle
import klepto
from preprocessing import *
from sklearn.model_selection import train_test_split
from pyke import kepconvert

def LoadDataset(dataset_name='lc_original.csv', directory='data/'):

    path = directory + dataset_name

    X = None
    y = None

    try:
        if (os.path.isfile(str(path))):
            print('Dataset: ' + dataset_name + ' found. Loading...')
            data = pd.read_csv(str(path))
            y = data.LABEL
            X = data.drop('LABEL', axis=1)
    except:
        print('Dataset: ' + dataset_name + ' not found. ')

    return X, y

def LoadPickledData(dataset_name='output.pkl',directory='C:\\Users\\DYN\\Desktop\\exoplanet_classification_repo\\data\\pickled_data\\'):

    d = klepto.archives.dir_archive(directory + 'transit_data_train', cached=True, serialized=True)

    #pvals_data = pickle.load(open(directory + 'K_keys/' + dataset_name, 'rb'))
    #transits_data = pickle.load(open(directory + 'K_results/' + dataset_name, 'rb'))
    #null_data = pickle.load(open(directory + 'K_time/' + dataset_name, 'rb'))

    d.load('keys')
    d.load('results')
    d.load('time')
    keys = d['keys']
    time = d['time']

    data = d['results']

    # convert to numpy array fo float type from object type
    pvals = np.array( list((data[:, 0])), dtype=np.float)
    transits = np.array( list((data[:, 1])), dtype=np.float)
    null = np.array( list((data[:, 2])), dtype=np.float)

    X = np.vstack([transits, null])
    y = np.hstack([np.ones(transits.shape[0]), np.zeros(null.shape[0])])


    #if categorical: y = np_utils.to_categorical(y, np.unique(y).shape[0])
    #if whiten: X = preprocessing.scale(X, axis=1)

    return X, y, pvals, keys, time


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
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state= 42, stratify=y_train)

    #X_train =  X_train.as_matrix().astype(np.float)

    if val_set:
        X_val = X_val.as_matrix().astype(np.float)

    #X_test =  X_test.as_matrix().astype(np.float)

    if val_set:
        return (X_train, y_train, X_val, y_val, X_test, y_test)
    else:
        return (X_train, y_train, X_test, y_test)

def ReadFITS():

    file = kepconvert('data\\kplr000757076-2011271113734_INJECTED-inj1_llc.fits.gz', columns='TIME,SAP_FLUX,PDCSAP_FLUX,SAP_FLUX_ERR,SAP_QUALITY', outfile='data\\kplr000757076-2011271113734_INJECTED-inj1_llc.csv',conversion='fits2csv')


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
            X_processed = handler.fillNaN(fillValue=0.0)
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

    # #X, y = LoadDataset('lc_original.csv')
    # X_train, y_train, pvals, keys, time = LoadPickledData()
    #
    #
    # y = pd.DataFrame(y_train, columns=['LABEL'])
    # X = pd.DataFrame(X_train)
    #
    # new_df = y.join(X)
    #
    # new_df = new_df.iloc[0:1000, :]
    #
    # new_df.to_csv('data/shortedX.csv')

    ReadFITS()


    #
    # X.to_csv('F:\X_Test.csv')
    # y.to_csv('F:\y_test.csv')

    # X_train = pd.read_csv('F:\X_Test.csv')
    # y_train = pd.read_csv('F:\y_test.csv')
    #
    # print('')
    #
    # X_train =  X_train.drop(['0'], axis=1 )
    # y_train = y_train.drop(['0'], axis=1)
    #
    # X_train_shortened = X_train[10000]
    # y_train_shortened = y_train[10000]
    #
    #
    # print('')
    # # Read filename and store lines in list
    # filenames = list(open('data/datasets.txt'))
    #
    # for fn in filenames:
    #     words = fn.rstrip('\n').split("_")
    #
    #     GenerateDataset(X, y, fn.rstrip('\n'), words)

