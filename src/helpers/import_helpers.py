import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import klepto
from sklearn.preprocessing import StandardScaler
from definitions import OUTPUT_DIR


def load_dataset(dataset_name=''):
    """
    Loads a CSV dataset and splits it into features and target columns.

    Args:
        dataset_name: Name of dataset to load

    Returns:
        X (features), and y (target column)
    """

    path = os.path.join(OUTPUT_DIR, dataset_name)

    try:
        if os.path.isfile(path):
            print(f'Dataset: {dataset_name} found. Loading...')
            # Load dataset (example path)
            df = pd.read_csv(path)

            # Split into features and labels
            X = df.drop('LABEL', axis=1).values
            y = df['LABEL'].values

            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            return X_train, X_test, y_train, y_test
        else:
            print(f'Dataset: {dataset_name} not found.')
            return None, None, None, None
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return None, None, None, None


def load_pickled_dataset(dataset_name='', directory='data/processed/'):
    """
    Loads a pickled dataset and splits it into features and target columns.

    Args:
        dataset_name: Name of dataset to load
        directory: Directory of the dataset

    Returns:
        X (features), and y (target column)
    """

    path = os.path.join(directory, dataset_name)

    try:
        if os.path.isfile(path):
            print(f'Dataset: {dataset_name} found. Loading...')
            data = pd.read_pickle(path)
            y = data['LABEL']
            X = data.drop('LABEL', axis=1)
            return X, y
        else:
            print(f'Dataset: {dataset_name} not found.')
            return None, None
    except Exception as e:
        print(f"An error occurred while loading the pickled dataset: {e}")
        return None, None


def load_large_pickled_data(dataset_name='output.pkl', directory='data/pickled_data/'):
    """
    Loads large pickled data using Klepto.

    Args:
        dataset_name: Name of dataset to load
        directory: Directory of the dataset

    Returns:
        X, y, pvals, keys, time (loaded data)
    """

    path = os.path.join(directory, dataset_name)

    try:
        d = klepto.archives.dir_archive(path, cached=True, serialized=True)
        d.load('keys')
        d.load('results')
        d.load('time')
        keys = d['keys']
        time = d['time']

        data = d['results']

        # Convert to numpy array from object type
        pvals = np.array([data[:, 0]], dtype=np.float)
        transits = np.array([data[:, 1]], dtype=np.float)
        null = np.array([data[:, 2]], dtype=np.float)

        X = np.vstack([transits, null])
        y = np.hstack([np.ones(transits.shape[0]), np.zeros(null.shape[0])])

        return X, y, pvals, keys, time
    except Exception as e:
        print(f"An error occurred while loading large pickled data: {e}")
        return None, None, None, None, None


def split_data(X, y, test_size=0.20, val_set=False):
    """
    Splits the data into training and testing sets, and optionally a validation set.

    Args:
        X: Features matrix to split
        y: Target column to split
        test_size: Size of the splits in percentage
        val_set: Boolean to determine if validation set is also created

    Returns:
        X_train, y_train, X_test, y_test (and optionally X_val, y_val)
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    if val_set:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42,
                                                          stratify=y_train)
        return X_train.astype(np.float), y_train, X_val.astype(np.float), y_val, X_test.astype(np.float), y_test
    else:
        return X_train.astype(np.float), y_train, X_test.astype(np.float), y_test
