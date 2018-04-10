import configparser
import time
import json
import os

from keras import backend as K

from data import *
from model import *
from evaluate import ModelEvaluator
from optimize import run_trials
from train import train_lstm

def load_model(name):
    root = 'models/'
    # Load json configuration
    json_file = open(root + name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # Load weights
    #model = model.load_weights(root + name + '.h5')

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




    # # # Train network
    # if (nn_type == 'LSTM'):
    #     train_lstm()
    # # #
    # # elif (nn_type == 'CNN'):
    # #     train_cnn()
    #

    best_run, best_model = run_trials()

    best_lstm = CNN_Model()
    best_lstm.SetModel(best_model)
    best_lstm.SaveModel()

    best_cnn = load_model('optim_65acc')
    best_cnn.SetSequenceLength()

    best_cnn.Compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1, momentum=0.25, decay=0.0001, nesterov=True))


    # Evaluate the best model on all (most) variations of the original dataset using cross validation.
    filenames = list(open('data/datasets.txt'))
    for fn in filenames:

        print ('Evaluating best model on dataset: ' + fn)

        fn_full = fn.rstrip('\n') + '.csv'
        X, y = LoadDataset(fn_full)

        X_train, y_train, X_test, y_test = SplitData(X, y)
        #X_train, y_train, X_val, y_val, X_test, y_test = SplitData(X,y, test_size=0.2, val_set=True)

        # Reshape data to 3D input
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        #X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        best_cnn.SetOutputDimension(1)
        best_cnn.SetSequenceLength(X_train.shape[1])

        hist = best_cnn.FitData(X_train, y_train, batch_size=32, nb_epochs=40, verbose=2)
        #hist = best_cnn.FitDataWithValidation(X_train, y_train, X_val, y_val, batch_size=32, nb_epochs=40, verbose=2)

        evaluator = ModelEvaluator(best_cnn, X_test=X_test, y_test=y_test, batch_size=32, generate_plots=True)

        #evaluator.PlotTrainingPerformance(hist)

        print (' --------------------------------------- ')


    print ('Finished evaluating the best model... Check results')





