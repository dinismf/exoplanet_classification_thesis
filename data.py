import os
import platform
import pandas as pd
import numpy as np
import pickle
import klepto
from preprocessing import *
from sklearn.model_selection import train_test_split
from pyke import *
import matplotlib.pyplot as plt
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

    X_train =  X_train.as_matrix().astype(np.float)

    if val_set:
        X_val = X_val.as_matrix().astype(np.float)

    X_test =  X_test.as_matrix().astype(np.float)

    if val_set:
        return (X_train, y_train, X_val, y_val, X_test, y_test)
    else:
        return (X_train, y_train, X_test, y_test)

def ReadFITS():

    #file = kepconvert('data\\fits\\kplr010471515-2009131105131_llc.fits', columns='TIME,SAP_FLUX,PDCSAP_FLUX,SAP_FLUX_ERR,SAP_QUALITY', outfile='data\\kplr000757076-2011271113734_INJECTED-inj1_llc.csv',conversion='fits2csv')


    # Read Original FITS into Light Curve structure
    og_lc = KeplerLightCurveFile( path='data\\fits\\kplr010027247-2012179063303_llc.fits')
    print(og_lc.header())
    og_lc_pdcsap = og_lc.get_lightcurve('PDCSAP_FLUX')

    print(og_lc_pdcsap.keplerid)
    flattened_lc = og_lc_pdcsap.flatten()

    # Detect best period
    #postlist, trial_periods, best_period = box_period_search(flattened_lc, nperiods=2000)
    #print('Best period: ', best_period)


    # Fold light curve
    folded_lc = flattened_lc.fold(period=0.868295, phase=2455022.262)

    binned_lc = folded_lc.bin(binsize=100, method='median')

    plt.plot(binned_lc.time, binned_lc.flux, 'x', markersize=1, label='FLUX')
    plt.show()
    # kephead('data\\fits\\kplr010027247-2012179063303_llc.fits', outfile='kplr010027247-2012179063303_llc_HEAD.txt', keyname='mag')
    # kepdraw('data\\fits\\kplr010027247-2012179063303_llc.fits', plottype='pretty', datacol='PDCSAP_FLUX')
    # #
    # kepflatten(infile='data\\fits\\kplr010027247-2012179063303_llc.fits', outfile='data\\fits\\kplr010027247-2012179063303_llcFLATTENED.fits', nsig=3, stepsize=1, npoly=2,niter=10, overwrite=True)
    # kepdraw('data\\fits\\kplr010027247-2012179063303_llcFLATTENED.fits', outfile=None, plottype='pretty', datacol='DETSAP_FLUX')
    #
     #kepfold(infile='data\\fits\\kplr010027247-2012179063303_llcFLATTENED.fits', outfile='data\\fits\\kplr010027247-2012179063303_llcFOLDED.fits',period=0.653534, bjd0=2455372.883, bindata=True, nbins=100)
    # # kepdraw('data\\fits\\kplr010471515-2010078095331_llc_FOLDED.fits', plottype='pretty')
    print()

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

        # d = klepto.archives.dir_archive('pickle_data/transit_data_train', cached=True, serialized=True)
        #
        # d['keys'] = data.keys
        # d['results'] = data.results
        # d['time'] = data.t
        #
        # d.dump()
        # d.clear()




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




    # X, y = LoadDataset('lc_original.csv')
    # # Read filename and store lines in list
    # filenames = list(open('data/datasets.txt'))
    #
    # for fn in filenames:
    #     words = fn.rstrip('\n').split("_")
    #
    #     GenerateDataset(X, y, fn.rstrip('\n'), words)

