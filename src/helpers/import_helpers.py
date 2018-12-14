import os
import klepto
import pandas as pd
from sklearn.model_selection import train_test_split
# from pyke import *


def LoadDataset(dataset_name='', directory='data/processed/'):
    """

    Args:
        dataset_name: Name of dataset to load
        directory: Directory of the dataset

    Returns: Returns X (features), and y (target column)

    """

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

def LoadPickledDataset(dataset_name='', directory='data/processed/'):
    """

    Args:
        dataset_name: Name of dataset to load
        directory: Directory of the dataset

    Returns:

    """

    path = directory + dataset_name

    X = None
    y = None

    try:
        if (os.path.isfile(str(path))):
            print('Dataset: ' + dataset_name + ' found. Loading...')
            data = pd.read_pickle(str(path))
            y = data.LABEL
            X = data.drop('LABEL', axis=1)
    except:
        print('Dataset: ' + dataset_name + ' not found. ')

    return X, y


def LoadLargePickledData(dataset_name='output.pkl',directory='C:\\Users\\DYN\\Desktop\\exoplanet_classification_repo\\data\\pickled_data\\'):
    """
     Loads large pickled data using Klepto

    Args:
        dataset_name: Name of dataset to load
        directory: Directory of the dataset

    Returns:

    """

    d = klepto.archives.dir_archive(directory + dataset_name, cached=True, serialized=True)

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

    return X, y, pvals, keys, time


def SplitData(X, y, test_size=0.20, val_set = False):
    """
    Splits X and y into training and testing sets, and optionally a validation set also.

    Args:
        X: Features matrix to split
        y: Target column to split
        test_size: Size of the splits in percentage
        val_set: Boolean to determine if validation set is also created.

    Returns:
        X_train = Training features
        y_train = Training target column
        X_test = Testing features
        y_test = Testing target column
        X_val (Optional) = Validation features
        y_val (Optional) = Validation target column

    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42, stratify=y)

    if val_set:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state= 42, stratify=y_train)

    X_train =  X_train.as_matrix().astype(np.float)

    if val_set:
        X_val = X_val.as_matrix().astype(np.float)

    X_test =  X_test.as_matrix().astype(np.float)

    if val_set:
        return (X_train, y_train, X_val, y_val, X_test, y_test)
    else:
        return (X_train, y_train, X_test, y_test)


# def GenerateDataset(og_X, og_y, filename, filename_words):
#     """
#     Generates multiple pre_processed versions of an input dataset
#
#     Args:
#         og_X: Original features data
#         og_y: Original target column data
#         filename: Name of the dataset
#         filename_words: Datasets to generate
#     """
#
#     root = 'data/'
#     cadence = filename_words[0]
#     scaling = filename_words[1]
#
#     try:
#         nan_handling = filename_words[2]
#     except:
#         nan_handling = None
#
#     try:
#         oversampling = filename_words[3]
#     except:
#         oversampling = None
#
#
#
#     if (os.path.isfile(root + filename + '.csv')):
#         print('Dataset: ' + filename + ' already exists')
#     else:
#
#         X_processed = og_X
#         y_processed = og_y
#
#         # Standardize of Normalise the data
#         if (scaling == 'std'):
#             X_processed = Standardizer().standardize(og_X, na_values=True)
#
#         elif (scaling == 'norm'):
#
#             X_processed = Normalizer().normalize(og_X, na_values=True)
#
#         # Handle the missing values in the timeseries data
#         handler = MissingValuesHandler(X_processed)
#
#         if (nan_handling == 'nanmasked'):
#             X_processed = handler.fillNaN(fillValue=0.0)
#         elif (nan_handling == 'nanremoved'):
#             X_processed = handler.removeNaN()
#
#         elif (nan_handling == 'nanimputed'):
#             X_processed = handler.imputeNaN()
#
#         elif (nan_handling == 'naninterpolated'):
#             nan_handling = None
#         elif (nan_handling == 'nanarima'):
#             nan_handling = None
#
#         # Oversample
#         if (oversampling == 'SMOTE' and nan_handling is not None):
#             X_processed, y_processed = Oversampler().OversampleSMOTE(X_processed, og_y)
#         elif (oversampling == 'ADASYN' and nan_handling is not None):
#             X_processed, y_processed = Oversampler().OversampleADASYN(X_processed, og_y)
#
#         # Merge new dataframe
#         y_processed = pd.DataFrame(y_processed, columns=['LABEL'])
#         new_df = y_processed.join(X_processed)
#
#         # Output the processed dataframe to .csv
#         path = str (root + filename + '.csv')
#         new_df.to_csv(path, na_rep='nan', index=False)
#         print('Dataset: ' + filename + ' created successfully')
