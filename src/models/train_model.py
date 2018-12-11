from src.models.model import *
from src.models.optimize import *
from src.models.train import train_cnn_cv, train_lstm_cv


def optimize_models(config):

    # Load the split training data (used during optimization of weights and hyperparameters) and unseen testing data.
    X_train_loaded = pd.read_pickle('kepler//testing_data//lstm_binneddata_XTRAIN.pkl')
    y_train_loaded = pd.read_pickle('kepler//testing_data//lstm_binneddata_YTRAIN.pkl')
    X_test_loaded = pd.read_pickle('kepler//testing_data//lstm_binneddata_XTEST.pkl')
    y_test_loaded = pd.read_pickle('kepler//testing_data//lstm_binneddata_YTEST.pkl')

    X_train_loaded = X_train_loaded.as_matrix().astype(np.float)
    X_test_loaded = X_test_loaded.as_matrix().astype(np.float)

    X_train_loaded = X_train_loaded.reshape(X_train_loaded.shape[0], X_train_loaded.shape[1], 1)
    X_test_loaded = X_test_loaded.reshape(X_test_loaded.shape[0], X_test_loaded.shape[1], 1)


    models = config['models']

    for model in range(len(models)):

        if model == "LSTM":

            best_model_config = run_trials('lstm', evals=10)


            print('Final Evaluation of Best Model Configuration')
            print('\n')

            best_lstm = LSTM_Model(output_dim=1, sequence_length=X_train_loaded.shape[1])
            best_lstm.LoadLSTMConfiguration(best_model_config)

            #best_lstm = LSTM_Model(output_dim=1, sequence_length=X_train_loaded.shape[1], nb_lstm_layers=0, nb_lstm_units=10, nb_fc_layers=2, nb_fc_units=64, dropout=0.29796647089233186,
            #                       activation='prelu', batch_normalisation=True)
            best_lstm.Build()
            best_lstm.Compile(loss='binary_crossentropy',
                        optimizer=SGD(lr=0.001 * 1.64341939565237, momentum=0.25, decay=0.0001,
                                      nesterov=True), metrics=['accuracy'])

            train_lstm_cv(best_lstm, X_train_loaded, y_train_loaded, X_test_loaded, y_test_loaded, nb_cv=5, batch_size=16, nb_epochs=43,
                             save_name='LSTM')

        elif model == "CNN":

            best_model_config = run_trials('cnn', evals=10)

            print('Final Evaluation of Best Model Configuration')
            print('\n')

            best_cnn = CNN_Model(output_dim=1, sequence_length=X_train_loaded.shape[1])
            best_cnn.LoadCNNConfiguration(best_model_config)

            # best_lstm = LSTM_Model(output_dim=1, sequence_length=X_train_loaded.shape[1], nb_lstm_layers=0, nb_lstm_units=10, nb_fc_layers=2, nb_fc_units=64, dropout=0.29796647089233186,
            #                       activation='prelu', batch_normalisation=True)
            best_cnn.Build()
            best_cnn.Compile(loss='binary_crossentropy',
                              optimizer=SGD(lr=0.001 * 1.64341939565237, momentum=0.25, decay=0.0001,
                                            nesterov=True), metrics=['accuracy'])

            train_cnn_cv(best_cnn, X_train_loaded, y_train_loaded, X_test_loaded, y_test_loaded, nb_cv=10,
                          batch_size=16, nb_epochs=43,
                          save_name='CNN')



def train_models(config):


def main():










