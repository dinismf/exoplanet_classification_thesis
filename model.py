import numpy as np
import time
from keras.models import Sequential, Model
from keras.layers import Input, Masking, Dense, Activation, Dropout, LSTM, Conv1D, Flatten, GlobalAveragePooling1D, BatchNormalization, MaxPool1D
from keras.optimizers import Adam, SGD


class Keras_Model:
    pass

class RNN_Model:
    pass

class LSTM_Model:

    def __init__(self, nb_layers, nb_units, output_dim, sequence_length, dropout, activation, maskingvalue):

        print('Initialising LSTM Model...')

        self.model = Sequential()

        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.dropout = dropout
        self.activation = activation

        # Value to mask from the input sequences
        self.maskingvalue = maskingvalue


    def Build(self):
        self.AddInputLayer()
        self.AddDropoutLayer()
        self.AddLSTMLayer()
        self.AddDropoutLayer()
        self.AddOutputLayer()

    def Compile(self, loss, optimizer, metrics):
        start = time.time()
        self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        print("Compilation Time: ", time.time() - start)
        print(self.model.summary())

    def AddMaskingLayer(self, value):
        # Masking layer
        self.model.add(Masking(mask_value=value, input_shape=(self.sequence_length, self.output_dim)))

    def AddInputLayer(self, return_seqs=True):
        # https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
        # Add masking layer is masking value exists
        if (self.maskingvalue is not None):
            self.AddMaskingLayer(self.maskingvalue)
            self.model.add(LSTM(self.nb_units, return_sequences=return_seqs))
        else:
            self.model.add(LSTM(self.nb_units, input_shape=(self.sequence_length, self.output_dim), return_sequences=return_seqs))


    def AddDropoutLayer(self):
        self.model.add(Dropout(self.dropout))

    def AddLSTMLayer(self):
        self.model.add(LSTM(self.nb_units))

    def AddOutputLayer(self):
        self.model.add(Dense(self.output_dim))
        self.model.add(Activation('sigmoid'))

    def FitData(self, X_train, y_train, batch_size, nb_epochs):
        self.model.fit(X_train, y_train, batch_size, nb_epochs)

    def Evaluate(self, X_test, y_test, batch_size):
        score, acc = self.model.evaluate(X_test, y_test, batch_size)

        print('Test score:', score)
        print('Test accuracy:', acc)


    def Predict(self, X_test, y_test):
        Y_score = self.model.predict(X_test)
        Y_predict = np.argmax(Y_score, axis=1)
        Y_true = np.argmax(y_test,axis=1)

        return (Y_score, Y_predict, Y_true)



class CNN_Model:

    def __init__(self, output_dim, sequence_length, x_trainshape):

        print('Initialising LSTM Model...')

        self.model = Sequential()

        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.xtrainshape = x_trainshape

    def Build(self):
        # self.AddInputLayer(filters=128, kernel_size=8)
        # self.AddCNNBlock(filters=256, kernel_size=5)
        # self.AddCNNBlock(filters=128, kernel_size=3)
        # self.model.add(GlobalAveragePooling1D())
        # self.AddOutputLayer()
        self.model.add(Conv1D(filters=8, kernel_size=11, activation='relu', input_shape=self.xtrainshape[1:]))
        self.model.add(MaxPool1D(strides=4))
        self.model.add(BatchNormalization())
        self.model.add(Conv1D(filters=16, kernel_size=11, activation='relu'))
        self.model.add(MaxPool1D(strides=4))
        self.model.add(BatchNormalization())
        self.model.add(Conv1D(filters=32, kernel_size=11, activation='relu'))
        self.model.add(MaxPool1D(strides=4))
        self.model.add(BatchNormalization())
        self.model.add(Conv1D(filters=64, kernel_size=11, activation='relu'))
        self.model.add(MaxPool1D(strides=4))
        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

    def Compile(self, loss, optimizer, metrics):
        start = time.time()
        self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        print("Compilation Time: ", time.time() - start)
        print(self.model.summary())

    def AddInputLayer(self, filters, kernel_size):
        self.model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(self.sequence_length, self.output_dim)))
        self.model.add(BatchNormalization())

    def AddBatchNormLayer(self):
        self.model.add(BatchNormalization())

    def AddCNNBlock(self, filters, kernel_size):
        self.model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
        self.AddBatchNormLayer()

    def AddOutputLayer(self):
        self.model.add(Dense(self.output_dim))
        self.model.add(Activation('sigmoid'))

    def FitData(self, X_train, y_train, batch_size, nb_epochs,cb1=None,cb2=None):
        return self.model.fit(X_train, y_train, batch_size, nb_epochs, verbose=1, callbacks=[cb1])

    def FitDataWithValidation(self, X_train, y_train, X_val, y_val, batch_size, nb_epochs,cb1=None,cb2=None):
        return self.model.fit(X_train, y_train, batch_size, nb_epochs, verbose=1, validation_data=(X_val, y_val), callbacks=[cb1])

    def Evaluate(self, X_test, y_test, batch_size):
        score, acc = self.model.evaluate(X_test, y_test, batch_size)

        print('Test score:', score)
        print('Test accuracy:', acc)


    def Predict(self, X_test, y_test):
        Y_score = self.model.predict(X_test)
        Y_predict = self.model.predict_classes(X_test)
        #Y_predict = Y_score.argmax(axis=-1)
       # Y_predict = np.argmax(Y_score, axis=1)
        Y_true = np.argmax(y_test,axis=1)

        return (Y_score, Y_predict, Y_true)
    pass