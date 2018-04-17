import configparser
import time
import json
import os

from keras import backend as K

from data import *
from model import *
from evaluate import ModelEvaluator
#from optimize import run_trials
from train import train_lstm, train_cnn

def load_model(name, weights=True):
    root = 'models/final/'
    # Load json configuration
    json_file = open(root + name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # Load weights
    if weights:
        model.load_weights(root + name + '.h5')

    cnn = CNN_Model(output_dim=1, sequence_length=10)
    cnn.SetModel(model=model)

    return cnn


if __name__ == '__main__':

    print(K.tensorflow_backend._get_available_gpus())
    config_filename = 'experiment_lstm.ini'
    #config_filename = 'experiment_cnn.ini'

    if not os.path.isfile(config_filename):
        raise Exception('File does not exist')

    # Read the experiment configuration
    config = configparser.ConfigParser()
    config.read(config_filename)

    # Retrieve network parameters
    nn_type = config.get('Network', 'nn_type')
   # nb_layers = config.get('Network', 'nb_layers')
    nb_hidden = config.get('Network', 'hidden_units')
    activation = config.get('Network', 'activation')
    dropout = config.get('Network', 'dropout')

    # Retrieve training parameters

    # Retrieve other options
    root_filename = config.get('Options', 'filename')
    plot_loss = config.getboolean('Options', 'plot_loss')
    save_model = config.getboolean('Options', 'save_model')


    # best_model_config, X_train, y_train, X_test, y_test = run_trials('cnn')
    #
    # # Split training data to train and validation splits
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42,
    #                                                   stratify=y_train)
    # # Reshape data to 3D input
    # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    # X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    #
    #
    # best_cnn = CNN_Model(output_dim=1, sequence_length=X_train.shape[1])
    # best_cnn.LoadConfiguration(best_model_config)
    #
    # hist = best_cnn.FitDataWithValidation(X_train, y_train, X_val, y_val, batch_size=best_model_config['batch_size'], nb_epochs=40, verbose=2)
    #
    # evaluator = ModelEvaluator(best_cnn, X_test=X_test, y_test=y_test, batch_size=best_model_config['batch_size'], generate_plots=True)
    #
    # evaluator.PlotTrainingPerformance(hist)
    #
    # best_cnn.SaveModel('best_cnn')


    print('Loading the pickled data...')
    X_train, y_train, pvals, keys, time = LoadPickledData()
    print('Loaded the pickled data...')

    X_test,y_test = LoadDataset('lc_std_nanimputed_shortened.csv')

    #X_train = X_train[0:1000]
    #y_train = y_train[0:1000]

    # Standardize training data
    X_train = Standardizer().standardize(X_train, na_values=False)

    print('Spliting training data into train and validation')

    # Split data
    X_train, y_train, X_val, y_val = SplitData(X_train,y_train, test_size=0.2)
    X_test = X_test.as_matrix().astype(np.float)

    #X_train, y_train, X_val, y_val, X_test, y_test = SplitData(X,y, test_size=0.2, val_set=True)


    print('Reshaping the input data to 3D')
    # Reshape data to 3D input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    #y_train = y_train.reshape(y_train.shape[0], 1)
    #y_test = y_test.reshape(y_test.shape[0], 1)

    # best_cnn = CNN_Model(nb_blocks=2, filters=16, kernel_size=5, activation='prelu', pooling='max', pool_strides=2, batch_norm=True, dense_units=64, conv_dropout=0.1, fc_dropout=0.3)
    # best_cnn.SetSequenceLength(X_train.shape[1])
    # best_cnn.SetOutputDimension(1)
    # best_cnn.Build()
    # best_cnn.Compile(loss='binary_crossentropy', optimizer=SGD(lr=0.01, momentum=0.25, decay=0.0001, nesterov=True))
    # train_cnn(best_cnn, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=16, nb_epochs=20, save=True)


    best_cnn = load_model('new_cnn', weights=True)
    #best_cnn.Compile(loss='binary_crossentropy', optimizer=SGD(lr=0.01, momentum=0.25, decay=0.0001, nesterov=True))
    ModelEvaluator(best_cnn, X_test=X_test, y_test=y_test, batch_size=16, segmentEval=True)


    # # Evaluate the best model on all (most) variations of the original dataset using cross validation.
    # filenames = list(open('data/datasets.txt'))
    # for fn in filenames:
    #
    #     print ('Evaluating best model on dataset: ' + fn)
    #
    #     fn_full = fn.rstrip('\n') + '.csv'
    #     X, y = LoadDataset(fn_full)
    #
    #     X_train, y_train, X_test, y_test = SplitData(X, y)
    #     #X_train, y_train, X_val, y_val, X_test, y_test = SplitData(X,y, test_size=0.2, val_set=True)
    #
    #     # Reshape data to 3D input
    #     X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    #     #X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    #     X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    #
    #     best_cnn.SetOutputDimension(1)
    #     best_cnn.SetSequenceLength(X_train.shape[1])
    #
    #     hist = best_cnn.FitData(X_train, y_train, batch_size=32, nb_epochs=40, verbose=2)
    #
    #     evaluator = ModelEvaluator(best_cnn, X_test=X_test, y_test=y_test, batch_size=32, generate_plots=True)
    #
    #     #evaluator.PlotTrainingPerformance(hist)
    #
    #     print (' --------------------------------------- ')


    print ('Finished evaluating the best model... Check results')








