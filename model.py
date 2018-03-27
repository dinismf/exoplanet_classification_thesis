import numpy as np
import time
from keras.models import Sequential, Model
from keras.layers import Input, Masking, Dense, Activation, Dropout, LSTM
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

        # Add masking layer is masking value exists
        if (self.maskingvalue is not None):
            self.AddMaskingLayer(self.maskingvalue)

        # Add input layer
        self.AddInputLayer(self.nb_units, self.output_dim, self.sequence_length)
        self.AddDropoutLayer(self.dropout)
        self.AddLSTMLayer(self.nb_units)
        self.AddDropoutLayer(self.dropout)

        self.AddOutputLayer(self.output_dim, self.activation)

    def Compile(self, loss, optimizer, metrics):
        start = time.time()
        self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        print("Compilation Time: ", time.time() - start)
        print(self.model.summary())

    def AddMaskingLayer(self, value):
        # Masking layer
        self.model.add(Masking(mask_value=value))

    def AddInputLayer(self, nb_units, output_dim, sequence_length, return_seqs=True,  ):
        # https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
        self.model.add(LSTM(nb_units, input_shape=(sequence_length, output_dim), return_sequences=return_seqs))

    def AddDropoutLayer(self, dropout):
        self.model.add(Dropout(dropout))

    def AddLSTMLayer(self, nb_units):
        self.model.add(LSTM(nb_units))

    def AddOutputLayer(self, output_dim, activation):
        self.model.add(Dense(output_dim))
        self.model.add(Activation(activation))

    def FitData(self, X_train, y_train, batch_size, nb_epochs):
        self.model.fit(X_train, y_train, batch_size, nb_epochs)

    def Evaluate(self, X_test, y_test, batch_size):
        self.model.evaluate(X_test, y_test, batch_size)

    def Predict(self, X_test, y_test):
        Y_score = self.model.predict(X_test)
        Y_predict = np.argmax(Y_score, axis=1)
        Y_true = np.argmax(y_test,axis=1)

        return (Y_score, Y_predict, Y_true)



class CNN_Model:
    pass