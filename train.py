from data import LoadData, SplitData
from preprocessing import *
from model import LSTM_Model, CNN_Model
from evaluate import ModelEvaluator
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ReduceLROnPlateau, EarlyStopping



def train_lstm(
               preprocessing_na = 'NA_MASKED', preprocessing_scale = 'STANDARDIZATION',
               batch_size = 32, nb_layers = 1, nb_hidden = 15, dropout = 0.5, nb_epochs = 5, masking_val = 0.0,
               optimizer = 'adam', metrics='accuracy',activation='sigmoid',
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

    X, y = Oversampler().OversampleSMOTE(X, y)
    X = pd.DataFrame(X)
    y = pd.Series(y)

    # Split data
    X_train, y_train, X_test, y_test = SplitData(X,y, test_size=0.2)
    #X_train, y_train, X_val, y_val, X_test, y_test = SplitData(X,y, test_size=0.2, val_set=True)

    # Reshape data to 3D input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    #X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    #y_train = y_train.reshape(y_train.shape[0], 1)
    #y_test = y_test.reshape(y_test.shape[0], 1)



    model = LSTM_Model(nb_layers=nb_layers, nb_units=nb_hidden, sequence_length=X_train.shape[1], output_dim=1, activation=activation, dropout=dropout, maskingvalue=None)
    model.Build()

    opt = SGD(lr=0.001)
    model.Compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)

    history = model.FitData(X_train, y_train, batch_size=batch_size, nb_epochs=nb_epochs)


    evaluator = ModelEvaluator(model, X_test=X_test, y_test=y_test, batch_size=batch_size)

    evaluator.PlotTrainingPerformance(history)




def train_cnn(preprocessing_na = 'NA_MASKED', preprocessing_scale = 'STANDARDIZATION',
               batch_size = 16, nb_epochs = 40,
               optimizer = 'adam', metrics='accuracy', masking_val=0.0,
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

    X, y = Oversampler().OversampleSMOTE(X, y)
    X = pd.DataFrame(X)
    y = pd.Series(y)


    # Split data
    #X_train, y_train, X_test, y_test = SplitData(X,y, test_size=0.2)
    X_train, y_train, X_val, y_val, X_test, y_test = SplitData(X,y, test_size=0.2, val_set=True)

    # Reshape data to 3D input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    #y_train = y_train.reshape(y_train.shape[0], 1)
    #y_test = y_test.reshape(y_test.shape[0], 1)


    model = CNN_Model(sequence_length=X_train.shape[1], output_dim=1, dropout=0.25, nb_blocks=2)
    model.Build()

    opt = SGD(lr=0.1, momentum=0.25, decay=0.0001, nesterov=True)
    #opt = Adam()
    #opt = RMSprop()
    model.Compile(loss='binary_crossentropy', optimizer=opt, metrics=metrics)


    reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001, verbose=1)
    #earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto')

    #hist = model.FitData(X_train, y_train, batch_size=batch_size, nb_epochs=nb_epochs, cb1=reduceLR)
    history = model.FitDataWithValidation(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, batch_size=batch_size, nb_epochs=nb_epochs)

    evaluator = ModelEvaluator(model, X_test=X_test, y_test=y_test, batch_size=batch_size)

    evaluator.PlotTrainingPerformance(history)





