import tensorflow as tf
from keras import backend as K
#from optimize import run_trials

from src.data.data import *
from src.models.model import *
from src.visualization.evaluate import ModelEvaluator
from src.models.train import train_cnn_cv


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

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


    ####################################################################################################################
    ######################################## Optimize and train CNN ############# ######################################
    ####################################################################################################################

    #best_model_config = run_trials('cnn', evals=50)

    ####################################################################################################################
    ####################################################################################################################

    ####################################################################################################################
    ######################################## Train CNN With CV ################## ######################################
    ####################################################################################################################

    # print('Loading the train and testing data...')
    #
    #
    # X_train_loaded, y_train_loaded = LoadDataset('lc_original.csv', directory='kepler//csv_data//OLD//')
    # X_train, y_train, X_test, y_test = SplitData(X_train_loaded, y_train_loaded, test_size=0.1)
    #
    # #Remove any NaNs
    # X_train = MissingValuesHandler(X_train).imputeNaN()
    # X_test = MissingValuesHandler(X_test).imputeNaN()
    #
    # # Standardize training data
    # X_train = Standardizer().standardize(X_train, na_values=False)
    # X_test = Standardizer().standardize(X_test, na_values=False)
    #
    #
    # # X_train_loaded = pd.read_pickle('kepler//testing_data//cnn_binneddata_XTRAIN.pkl')
    # # y_train_loaded = pd.read_pickle('kepler//testing_data//cnn_binneddata_YTRAIN.pkl')
    # # X_test_loaded = pd.read_pickle('kepler//testing_data//cnn_binneddata_XTEST.pkl')
    # # y_test_loaded = pd.read_pickle('kepler//testing_data//cnn_binneddata_YTEST.pkl')
    #
    # # X_train =  X_train.as_matrix().astype(np.float)
    # # X_test =  X_test.as_matrix().astype(np.float)
    # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    #
    # print('Final Evaluation of Best Model Configuration')
    # print('\n')
    #
    # #best_cnn = CNN_Model(output_dim=1, sequence_length=X_train_loaded.shape[1])
    # #best_cnn.LoadConfiguration(best_model_config)
    # best_cnn = CNN_Model(output_dim=1, sequence_length=X_train_loaded.shape[1], nb_blocks=1,
    #                      filters=128,
    #                      kernel_size=5,
    #                      activation='prelu', pooling='max', pool_size=3, pool_strides=2,
    #                      conv_dropout=0.08625213973083698, fc_dropout=0.2650803877994672,
    #                      dense_units=64, batch_norm=True)
    # best_cnn.Build()
    # best_cnn.Compile(loss='binary_crossentropy',
    #              optimizer=SGD(lr=0.001*1.64341939565237, momentum=0.25, decay=0.0001,
    #                            nesterov=True), metrics=['accuracy'])

    #train_cnn_cv(best_cnn, X_train_loaded, y_train_loaded, X_test_loaded, y_test_loaded, nb_cv=10, batch_size=best_model_config['batch_size'], nb_epochs=int(best_model_config['nb_epochs']),
    #             save_name='cnn_finalversion_50evals_dataset_globalbinned')
    train_cnn_cv(best_cnn, X_train, y_train, X_test, y_test, nb_cv=5, batch_size=32, nb_epochs=36)

    ####################################################################################################################
    ####################################################################################################################

    ####################################################################################################################
    ######################################## Optimize and train LSTM ############# ######################################
    ####################################################################################################################

    # #best_model_config = run_trials('lstm', evals=25)
    #
    # print('Loading the train and testing data...')
    # # X_train_loaded, y_train_loaded = LoadDataset('1st_binneddata_TRAIN.csv', directory='data//testing_data//')
    # # X_test_loaded, y_test_loaded = LoadDataset('1st_binneddata_TEST.csv', directory='data//testing_data//')
    # X_train_loaded = pd.read_pickle('kepler//testing_data//lstm_binneddata_XTRAIN.pkl')
    # y_train_loaded = pd.read_pickle('kepler//testing_data//lstm_binneddata_YTRAIN.pkl')
    # X_test_loaded = pd.read_pickle('kepler//testing_data//lstm_binneddata_XTEST.pkl')
    # y_test_loaded = pd.read_pickle('kepler//testing_data//lstm_binneddata_YTEST.pkl')
    #
    # X_train_loaded = X_train_loaded.as_matrix().astype(np.float)
    # X_test_loaded = X_test_loaded.as_matrix().astype(np.float)
    #
    # X_test_loaded = X_test_loaded.reshape(X_test_loaded.shape[0], X_test_loaded.shape[1], 1)
    #
    # print('Final Evaluation of Best Model Configuration')
    # print('\n')
    #
    # #best_lstm = LSTM_Model(output_dim=1, sequence_length=X_train_loaded.shape[1], nb)
    # #best_lstm.LoadLSTMConfiguration(best_model_config)
    #
    # best_lstm = LSTM_Model(output_dim=1, sequence_length=X_train_loaded.shape[1], nb_lstm_layers=0, nb_lstm_units=10, nb_fc_layers=2, nb_fc_units=64, dropout=0.29796647089233186,
    #                        activation='prelu', batch_normalisation=True)
    # best_lstm.Build()
    # best_lstm.Compile(loss='binary_crossentropy',
    #             optimizer=SGD(lr=0.001 * 1.64341939565237, momentum=0.25, decay=0.0001,
    #                           nesterov=True), metrics=['accuracy'])
    #
    # train_lstm_cv(best_lstm, X_train_loaded, y_train_loaded, X_test_loaded, y_test_loaded, nb_cv=5, batch_size=16, nb_epochs=43,
    #              save_name='lstm_finalversion_25evals_dataset_globalbinned')




    ####################################################################################################################
    ######################################## Train CNN/LSTM W/Out CV ########################################################
    ####################################################################################################################

    # X_train = pd.read_pickle('kepler//testing_data//cnn_binneddata_XTRAIN.pkl')
    # y_train= pd.read_pickle('kepler//testing_data//cnn_binneddata_YTRAIN.pkl')
    # X_test = pd.read_pickle('kepler//testing_data//cnn_binneddata_XTEST.pkl')
    # y_test = pd.read_pickle('kepler//testing_data//cnn_binneddata_YTEST.pkl')
    #
    # X_train, y_train, X_val, y_val = SplitData(X_train, y_train, test_size=0.1)
    #
    # X_test =  X_test.as_matrix().astype(np.float)
    # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    # X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    #best_cnn = CNN_Model(output_dim=1, sequence_length=X_train_loaded.shape[1])
    #best_cnn.LoadConfiguration(best_model_config)
    # best_cnn = CNN_Model(output_dim=1, sequence_length=X_train.shape[1], nb_blocks=1,
    #                      filters=128,
    #                      kernel_size=5,
    #                      activation='prelu', pooling='max', pool_size=3, pool_strides=2,
    #                      conv_dropout=0.08625213973083698, fc_dropout=0.2650803877994672,
    #                      dense_units=64, batch_norm=True)
    # best_cnn.Build()
    # best_cnn.Compile(loss='binary_crossentropy',
    #              optimizer=SGD(lr=0.001*1.64341939565237, momentum=0.25, decay=0.0001,
    #                            nesterov=True), metrics=['accuracy'])
    # train_cnn(best_cnn, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32, nb_epochs=36, test_eval=True)


    #best_lstm = LSTM_Model(output_dim=1, sequence_length=X_train_loaded.shape[1], nb)
    #best_lstm.LoadLSTMConfiguration(best_model_config)
    # best_lstm = LSTM_Model(output_dim=1, sequence_length=X_train.shape[1], nb_lstm_layers=0, nb_lstm_units=10, nb_fc_layers=2, nb_fc_units=64, dropout=0.29796647089233186,
    #                        activation='prelu', batch_normalisation=True)
    # best_lstm.Build()
    # best_lstm.Compile(loss='binary_crossentropy',
    #             optimizer=SGD(lr=0.001 * 1.64341939565237, momentum=0.25, decay=0.0001,
    #                           nesterov=True), metrics=['accuracy'])
    #
    # train_lstm(best_lstm, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=16, nb_epochs=43, test_eval=True)

    ####################################################################################################################
    ####################################################################################################################




    ####################################################################################################################
    ######################################## Load Pre-Trained Models and Evaluate ######################################
    ####################################################################################################################

    print('Loading the testing data...')
    X_test_loaded_cnn = pd.read_pickle('kepler//testing_data//cnn_binneddata_XTEST.pkl')
    y_test_loaded_cnn = pd.read_pickle('kepler//testing_data//cnn_binneddata_YTEST.pkl')
    X_test_loaded_lstm = pd.read_pickle('kepler//testing_data//lstm_binneddata_XTEST.pkl')
    y_test_loaded_lstm = pd.read_pickle('kepler//testing_data//lstm_binneddata_YTEST.pkl')

    X_test_loaded_cnn = X_test_loaded_cnn.as_matrix().astype(np.float)
    X_test_loaded_cnn = X_test_loaded_cnn.reshape(X_test_loaded_cnn.shape[0], X_test_loaded_cnn.shape[1], 1)
    X_test_loaded_lstm = X_test_loaded_lstm.as_matrix().astype(np.float)
    X_test_loaded_lstm = X_test_loaded_lstm.reshape(X_test_loaded_lstm.shape[0], X_test_loaded_lstm.shape[1], 1)

    print('Loading the Pretrained Models')
    print('\n')

    best_cnn = load_model('cnn_finalversion_50evals_dataset_globalbinned', weights=True)
    ModelEvaluator(best_cnn, X_test=X_test_loaded_cnn, y_test=y_test_loaded_cnn, batch_size=32)

    best_lstm = load_model('lstm_finalversion_25evals_dataset_globalbinned', weights=True)
    ModelEvaluator(best_lstm, X_test=X_test_loaded_lstm, y_test=y_test_loaded_lstm, batch_size=16)

    ###################################################################################################################
    ###################################################################################################################





