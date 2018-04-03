from data import LoadData, SplitData
from preprocessing import *
from model import LSTM_Model, CNN_Model
from sklearn.metrics import roc_auc_score, precision_score, recall_score, classification_report, confusion_matrix
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

def train_lstm(
               preprocessing_na = 'NA_MASKED', preprocessing_scale = 'STANDARDIZATION',
               batch_size = 10, nb_layers = 1, nb_hidden = 15, dropout = 0.5, nb_epochs = 20, masking_val = 0.0,
               optimizer = Adam(lr=0.01), metrics='accuracy',activation='sigmoid',
               save_model=False, plot_loss=False, plot_data=[], filename=''):


    # Load original data
    X, y = LoadData()



    # Or load pre-processed version of original
    if (preprocessing_na == 'NONE'):
        X = Standardizer().standardize(X, na_values=True)

    elif (preprocessing_na == 'NA_MASKED'):
        X = Standardizer().standardize(X, na_values=True)
        X = MissingValuesHandler(X).fillNaN(fillValue=masking_val)

    elif (preprocessing_na == 'NA_REMOVED'):
        X = MissingValuesHandler(X).removeNaN()
        X = Standardizer().standardize(X, na_values=True)

    elif (preprocessing_na == 'NA_ARIMA'):
        pass

    # Split data
    #X_train, y_train, X_test, y_test = SplitData(X,y, test_size=0.2)
    X_train, y_train, X_val, y_val, X_test, y_test = SplitData(X,y, test_size=0.2, val_set=True)

    # Reshape data to 3D input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    #y_train = y_train.reshape(y_train.shape[0], 1)
    #y_test = y_test.reshape(y_test.shape[0], 1)



    model = LSTM_Model(nb_layers=nb_layers, nb_units=nb_hidden, sequence_length=4767, output_dim=1, activation=activation, dropout=dropout, maskingvalue=masking_val)
    model.Build()

    opt = SGD(lr=0.001)
    model.Compile(loss='binary_crossentropy', optimizer=opt, metrics=metrics)

    model.FitData(X_train, y_train, batch_size=batch_size, nb_epochs=nb_epochs)

    model.Evaluate(X_test, y_test, batch_size)

    Y_score, Y_predict, Y_true = model.Predict(X_test, y_test)

    print('Classification Report: \n')
    print(classification_report(y_test, Y_predict))

    print('Confusion Matrix: \n')
    conf_matrix = confusion_matrix(y_test, Y_predict)
    print(conf_matrix)
    print('Done...')








def train_cnn(preprocessing_na = 'NA_MASKED', preprocessing_scale = 'STANDARDIZATION',
               batch_size = 32, nb_epochs = 50,
               optimizer = Adam(), metrics='accuracy', masking_val=0,
               save_model=False, plot_loss=False, plot_data=[], filename=''):
    # Load original data
    X, y = LoadData()

    # Or load pre-processed version of original
    if (preprocessing_na == 'NONE'):
        X = Standardizer().standardize(X, na_values=True)

    elif (preprocessing_na == 'NA_MASKED'):

        if (preprocessing_scale == 'NONE'):
            X = MissingValuesHandler(X).fillNaN(fillValue=masking_val)

        elif (preprocessing_scale == 'STANDARDIZATION'):
            X = Standardizer().standardize(X, na_values=True)
            X = MissingValuesHandler(X).fillNaN(fillValue=masking_val)

    elif (preprocessing_na == 'NA_REMOVED'):
        X = MissingValuesHandler(X).removeNaN()
        X = Standardizer().standardize(X, na_values=True)

    elif (preprocessing_na == 'NA_ARIMA'):
        pass

    # Split data
    X_train, y_train, X_test, y_test = SplitData(X,y, test_size=0.2)
    #X_train, y_train, X_val, y_val, X_test, y_test = SplitData(X,y, test_size=0.2, val_set=True)

    # Reshape data to 3D input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    #X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    #y_train = y_train.reshape(y_train.shape[0], 1)
    #y_test = y_test.reshape(y_test.shape[0], 1)

    test = X_train.shape[1:]

    model = CNN_Model(sequence_length=X_train.shape[1], output_dim=1, x_trainshape=X_train.shape)
    model.Build()

    opt = SGD(lr=0.001)
    model.Compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)


    reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001)
    #earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto')

    hist = model.FitData(X_train, y_train, batch_size=batch_size, nb_epochs=nb_epochs, cb1=reduceLR)
    #hist = model.FitDataWithValidation(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, batch_size=batch_size, nb_epochs=nb_epochs, cb1=reduceLR)

    #log = pd.DataFrame(hist.history)
    #print(log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_acc'])

    model.Evaluate(X_test, y_test, batch_size)

    Y_score, Y_predict, Y_true = model.Predict(X_test, y_test)

    print('Classification Report: \n')
    print(classification_report(y_test, Y_predict))

    print('Confusion Matrix: \n')
    conf_matrix = confusion_matrix(y_test, Y_predict)
    print(conf_matrix)
    print('Done...')






