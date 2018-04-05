import numpy as np
import time
from keras.models import Sequential, Model
from keras.layers import Input, Masking, Dense, Activation, Dropout,CuDNNLSTM, LSTM, Conv1D, Flatten, GlobalAveragePooling1D, BatchNormalization, MaxPool1D, AveragePooling1D
from keras.optimizers import Adam, SGD


class Keras_Model:
    def __init__(self):

        self.model = Sequential()

    def FitData(self, X_train, y_train, batch_size, nb_epochs):
        return self.model.fit(X_train, y_train, batch_size, nb_epochs)

    def FitDataWithValidation(self, X_train, y_train, X_val, y_val, batch_size, nb_epochs):
        return self.model.fit(X_train, y_train, batch_size, nb_epochs, verbose=1, validation_data=(X_val, y_val))

    def FitDataWithValidationCallbacks(self, X_train, y_train, X_val, y_val, batch_size, nb_epochs, cb1, cb2):
        return self.model.fit(X_train, y_train, batch_size, nb_epochs, verbose=1, validation_data=(X_val, y_val), callbacks=[cb1])

    def Evaluate(self, X_test, y_test, batch_size):
        score, acc = self.model.evaluate(X_test, y_test, batch_size)

        return score, acc

    def Predict(self, X_test, y_test):
        Y_score = self.model.predict(X_test)
        Y_predict = self.model.predict_classes(X_test)
        Y_true = np.argmax(y_test, axis=1)

        return (Y_score, Y_predict, Y_true)


class RNN_Model:
    pass

class LSTM_Model(Keras_Model):

    def __init__(self, nb_layers, nb_units, output_dim, sequence_length, dropout, activation, maskingvalue):
        Keras_Model.__init__(self)

        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.dropout = dropout
        self.activation = activation

        # Value to mask from the input sequences
        self.maskingvalue = maskingvalue

        if (self.maskingvalue is None):
            self.cuddn = True
        else:
            self.cuddn = False


    def Build(self):
        self.AddInputLayer()
        self.AddDropoutLayer()
        self.model.add(Flatten())
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
            self.model.add(CuDNNLSTM(self.nb_units, input_shape=(self.sequence_length, self.output_dim), return_sequences=return_seqs))


    def AddDropoutLayer(self):
        self.model.add(Dropout(self.dropout))

    def AddLSTMLayer(self):

        if self.cuddn:
            self.model.add(CuDNNLSTM(self.nb_units, return_sequences=True))
        else:
            self.model.add(LSTM(self.nb_units, return_sequences=True))


    def AddOutputLayer(self):
        self.model.add(Dense(self.nb_units))
        self.model.add(Dense(self.output_dim))
        self.model.add(Activation('sigmoid'))



class CNN_Model(Keras_Model):

    def __init__(self, output_dim, sequence_length,
                 nb_blocks=1,filters=8, kernel_size=5, activation = 'relu',
                 pooling='max', pool_strides = 3, pool_size = None,
                 dropout=0.5):

        print('Initialising CNN Model...')
        Keras_Model.__init__(self)

        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation

        self.pooling = pooling
        self.dropout = dropout

        if self.pooling == 'max':
            self.pooling_val = pool_strides
        elif self.pooling == 'average':
            self.pooling_val = pool_size

    def Build(self):

        self.AddInputBlock()

        for i in range(self.nb_blocks):
            self.AddCNNBlock()

        self.AddOutputBlock()

    def Compile(self, loss, optimizer, metrics):
        start = time.time()
        self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        print("Compilation Time: ", time.time() - start)
        print(self.model.summary())

    def AddInputBlock(self):

        self.model.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size, input_shape=(self.sequence_length, self.output_dim)))

        # Batch Norm Layer
        self.model.add(BatchNormalization())

        # Pooling Layer
        if (self.pooling == 'max'):
            self.model.add(MaxPool1D(strides=self.pooling_val))
        elif (self.pooling == 'average'):
            self.model.add(AveragePooling1D(pool_size=self.pooling_val))

        # Activation Layer
        self.model.add(Activation(self.activation))

        # Dropout layer
        if (self.dropout is not None):
            self.model.add(Dropout(self.dropout))




    def AddCNNBlock(self):

        # Convolution Layer
        self.model.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size))

        # Batch Norm Layer
        self.model.add(BatchNormalization())

        # Pooling Layer
        if (self.pooling == 'max'):
            self.model.add(MaxPool1D(strides=self.pooling_val))
        elif (self.pooling == 'average'):
            self.model.add(AveragePooling1D(pool_size=self.pooling_val))

        # Activation Layer
        self.model.add(Activation(self.activation))

        if (self.dropout is not None):
            self.model.add(Dropout(self.dropout))

    def AddOutputBlock(self):

        self.model.add(Flatten())
        self.model.add(Dense(64, activation=self.activation))

        if self.dropout is not None:
            self.model.add(Dropout(self.dropout))

        self.model.add(Dense(32, activation=self.activation))
        self.model.add(Dense(16, activation=self.activation))
        self.model.add(Dense(8, activation=self.activation))

        self.model.add(Dense(1, activation='sigmoid'))

