from data import *
from model import *
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from evaluate import ModelEvaluator
import json

def data():

    #X, y = LoadOriginalData()
    X, y = LoadDataset('lc_std_nanimputed.csv')
    #X, y = LoadDataset('lc_std_nanmasked_SMOTE.csv')
    #X, y = LoadDataset('lc_std.csv')

    # Split data
    #X_train, y_train, X_test, y_test = SplitData(X,y, test_size=0.2)
    X_train, y_train, X_val, y_val, X_test, y_test = SplitData(X, y, test_size=0.2, val_set=True)

    # Reshape data to 3D input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # y_train = y_train.reshape(y_train.shape[0], 1)
    # y_test = y_test.reshape(y_test.shape[0], 1)

    #return X_train, y_train, X_test, y_test
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_cnn_model(X_train, y_train, X_val, y_val, X_test, y_test):
#def create_cnn_model(X_train, y_train, X_test, y_test):

    cnn = CNN_Model(output_dim=1, sequence_length=X_train.shape[1],
                    nb_blocks={{choice([1,2,3])}}, filters={{choice([8, 16, 32, 64])}}, kernel_size={{choice([5, 8, 11])}},
                    activation={{choice(['prelu'])}}, pooling={{choice(['max', 'average'])}}, pool_size={{choice([1,2,3,4,5])}}, pool_strides={{choice([1,2,3,4,5])}},
                    dropout={{uniform(0,1)}})

    cnn.Build()

    cnn.Compile(loss='binary_crossentropy', optimizer=SGD(lr={{choice([0.1,0.01,0.001])}}, momentum={{choice([0., 0.25])}}, decay=0.0001, nesterov=True), metrics=['accuracy'])

    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001, verbose=1)
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto')

    # hist = model.FitData(X_train, y_train, batch_size=batch_size, nb_epochs=nb_epochs, cb1=reduceLR)
    batch_size = {{choice([16,32])}}

    #cnn.FitData(X_train=X_train, y_train=y_train,
    #                         batch_size=batch_size, nb_epochs=50, verbose=2, cb1=reduceLR, cb2=earlyStopping)
    cnn.FitDataWithValidationCallbacks(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                                           batch_size=batch_size, nb_epochs=1, verbose=2, cb1=reduceLR, cb2=earlyStopping)

    score, acc = cnn.Evaluate(X_test, y_test, batch_size, verbose=2)

    print('Test Accuracy: ', acc)

    return {'loss': -acc, 'status': STATUS_OK, 'model': cnn.GetModel()}


def create_lstm(X_train, y_train, X_test, y_test):

    lstm = LSTM_Model(output_dim=1, sequence_length=X_train.shape[1],
                    nb_lstm_layers={{choice([1, 2, 3])}}, nb_units={{choice([5, 10, 15])}},
                    activation={{choice(['relu', 'prelu'])}},
                    dropout={{uniform(0, 1)}})

    lstm.Build()

    lstm.Compile(loss='binary_crossentropy',
                optimizer={{choice([SGD(lr=0.1, momentum=0.25, decay=0.0001, nesterov=True)])}},
                metrics=['accuracy'])

    # reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001, verbose=1)

    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='auto')
    checkpointer = ModelCheckpoint(filepath='lstm_tuned.hdf5',
                                   verbose=0,
                                   save_best_only=True)


    # hist = model.FitData(X_train, y_train, batch_size=batch_size, nb_epochs=nb_epochs, cb1=reduceLR)
    batch_size = {{choice([16, 32, 64, 128])}}

    lstm.FitData(X_train=X_train, y_train=y_train, validation_split=0.08,
                batch_size=batch_size, nb_epochs=50, verbose=2, cb1=earlyStopping, cb2=checkpointer)
    # cnn.FitDataWithValidation(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
    #                                       batch_size=batch_size, nb_epochs=1, verbose=2)

    score, acc = lstm.Evaluate(X_test, y_test, batch_size, verbose=0)

    print('Test Accuracy: ', acc)

    evaluator = ModelEvaluator(lstm, X_test=X_test, y_test=y_test, batch_size=32, generate_plots=False)


    return {'loss': -acc, 'status': STATUS_OK, 'model': lstm.GetModel()}


def run_trials():

    start_time = time.time()

    best_run, best_model = optim.minimize(model=create_cnn_model, data=data, algo=tpe.suggest, max_evals=1, trials=Trials())

    print('Training time (seconds): ', (time.time() - start_time))

    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    json.dump(best_run, open("models/best_run.txt", 'w'))


    return best_run, best_model

