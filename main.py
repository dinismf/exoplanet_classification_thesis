import configparser
import time
import json
import os

from keras import backend as K
from optimize import run_trials

from data import *
from model import *
from evaluate import ModelEvaluator
from train import train_lstm, train_cnn, train_cnn_cv

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

    # CONFIG PARSER !

    # config_filename = 'experiment_lstm.ini'
    # #config_filename = 'experiment_cnn.ini'
    #
    # if not os.path.isfile(config_filename):
    #     raise Exception('File does not exist')

   #  # Read the experiment configuration
   #  config = configparser.ConfigParser()
   #  config.read(config_filename)
   #
   #  # Retrieve network parameters
   #  nn_type = config.get('Network', 'nn_type')
   # # nb_layers = config.get('Network', 'nb_layers')
   #  nb_hidden = config.get('Network', 'hidden_units')
   #  activation = config.get('Network', 'activation')
   #  dropout = config.get('Network', 'dropout')
   #
   #  # Retrieve training parameters
   #
   #  # Retrieve other options
   #  root_filename = config.get('Options', 'filename')
   #  plot_loss = config.getboolean('Options', 'plot_loss')
   #  save_model = config.getboolean('Options', 'save_model')
   #


    best_model_config = run_trials('cnn', evals=50)


    print('Reloading the train and testing data...')
    X_train_loaded, y_train_loaded = LoadDataset('1st_binneddata_TRAIN.csv', directory='data//testing_data//')
    #X_test_loaded, y_test_loaded = LoadDataset('1st_binneddata_TEST.csv', directory='data//testing_data//')
    X_train_loaded = pd.read_csv('data//testing_data//1st_binneddata_XTRAIN.csv', header=0)
    y_train_loaded = pd.read_csv('data//testing_data//1st_binneddata_YTRAIN.csv', header=0)
    X_test_loaded = pd.read_csv('data//testing_data//1st_binneddata_XTEST.csv', header=0)
    y_test_loaded = pd.read_csv('data//testing_data//1st_binneddata_YTEST.csv', header=0)

    X_train_loaded =  X_train_loaded.as_matrix().astype(np.float)
    X_test_loaded =  X_test_loaded.as_matrix().astype(np.float)

    X_test_loaded = X_test_loaded.reshape(X_test_loaded.shape[0], X_test_loaded.shape[1], 1)

    print('Final Evaluation of Best Model Configuration')
    print('\n')

    best_cnn = CNN_Model(output_dim=1, sequence_length=X_train_loaded.shape[1])
    best_cnn.LoadConfiguration(best_model_config)

    train_cnn_cv(best_cnn, X_train_loaded, y_train_loaded, X_test_loaded, y_test_loaded, nb_cv=10, batch_size=best_model_config['batch_size'], nb_epochs=int(best_model_config['nb_epochs']),
                 save_name='cnn_version2_50evals_dataset_conf_fps_binned')



    #
    # Train model on new data
    #
   #
   #  X, y = LoadDataset('binned_confirmed_fps_binned.csv')
   #  #X, y = LoadPickledDataset('folded_confirmed_candidates.pkl')
   #
   #  #X_train = X_train[0:1000]
   #  #y_train = y_train[0:1000]
   #
   #  #Split data
   #  print('Spliting training data into train and validation')
   #
   #  #X_test = X_test.as_matrix().astype(np.float)
   #
   #  #X_train, y_train, X_test, y_test = SplitData(X,y, test_size=0.2, val_set = False)
   #  X_train, y_train, X_val, y_val, X_test, y_test = SplitData(X,y, test_size=0.1, val_set=True)
   #
   #
   #  # Remove any NaNs
   #  X_train = MissingValuesHandler(X_train).imputeNaN()
   #  X_val = MissingValuesHandler(X_val).imputeNaN()
   #  X_test = MissingValuesHandler(X_test).imputeNaN()
   #
   #
   #  # Standardize training data
   #  X_train = Standardizer().standardize(X_train, na_values=False)
   #  X_val = Standardizer().standardize(X_val, na_values=False)
   #  X_test = Standardizer().standardize(X_test, na_values=False)
   #
   #
   #  # Save the test data
   #  y_test_save = pd.DataFrame(y_test, columns=['LABEL'])
   #  test_df = y_test_save.join(pd.DataFrame(X_test))
   #  test_df.to_csv('data//testing_data//1st_binneddata_test.csv')
   #
   #
   #  print('Reshaping the input data to 3D')
   #  # Reshape data to 3D input
   #  X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
   #  X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
   #  X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
   #
   #  best_cnn = CNN_Model()
   #  best_cnn.SetSequenceLength(X_train.shape[1])
   #  best_cnn.SetOutputDimension(1)
   #  best_cnn.LoadConfigurationFromFile(config_name='cnn_50evals_dataset_conf_fps_new')
   #  # best_cnn = CNN_Model(output_dim=1, sequence_length=X_train.shape[1], nb_blocks=3,
   #  #                     filters=8,
   #  #                     kernel_size=9,
   #  #                     activation='prelu', pooling='max', pool_size=2, pool_strides=2,
   #  #                     conv_dropout=1.623235288543759e-05, fc_dropout=0.1529870895215522,
   #  #                     dense_units=32, batch_norm=True)
   #
   #
   #  best_cnn.Build()
   #  best_cnn.Compile(loss='binary_crossentropy', optimizer=SGD(lr=0.001, momentum=0.20, decay=0.0001, nesterov=True))
   # # best_cnn.Compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=0.00000001))
   #
   #  train_cnn(best_cnn, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=16, nb_epochs=15, test_eval=True, save=False)
   #  #train_cnn_cv(best_cnn, X_train, y_train, X_test, y_test, nb_cv=10, batch_size=16, nb_epochs=30, save=False)







    ### Load weights and Evaluate Model

    #best_cnn = load_model('new_cnn', weights=True)
    #best_cnn.Compile(loss='binary_crossentropy', optimizer=SGD(lr=0.01, momentum=0.25, decay=0.0001, nesterov=True))
    #ModelEvaluator(best_cnn, X_test=X_test, y_test=y_test, batch_size=16, segmentEval=True)


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








