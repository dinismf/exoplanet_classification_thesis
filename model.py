import numpy as np
import time
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Masking, Dense, Activation, Dropout,CuDNNLSTM, LSTM, Conv1D, Flatten, GlobalAveragePooling1D, BatchNormalization, MaxPool1D, AveragePooling1D, PReLU
from keras.optimizers import Adam, SGD



class Keras_Model:
    def __init__(self):

        self.model = Sequential()

    def Compile(self, loss, optimizer, metrics='accuracy'):
        start = time.time()
        self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        print("Compilation Time: ", time.time() - start)
        print(self.model.summary())


    def FitData(self, X_train, y_train, batch_size, nb_epochs, cb1, cb2, verbose=1, validation_split = None):

        if validation_split is not None :
                hist = self.model.fit(X_train, y_train, batch_size, nb_epochs, validation_split=validation_split, verbose=verbose, callbacks=[cb1,cb2])
        else:
            hist = self.model.fit(X_train, y_train, batch_size, nb_epochs, verbose=verbose)

        return hist

    def FitDataWithValidation(self, X_train, y_train, X_val, y_val, batch_size, nb_epochs, verbose = 1):
        return self.model.fit(X_train, y_train, batch_size, nb_epochs, verbose=verbose, validation_data=(X_val, y_val))

    def FitDataWithValidationCallbacks(self, X_train, y_train, X_val, y_val, batch_size, nb_epochs, cb1, cb2, verbose):
        return self.model.fit(X_train, y_train, batch_size, nb_epochs, verbose=verbose, validation_data=(X_val, y_val), callbacks=[cb1, cb2])

    def Evaluate(self, X_test, y_test, batch_size, verbose=1):

        score, acc = self.model.evaluate(X_test, y_test, batch_size, verbose=verbose)
        return score, acc

    def Predict(self, X_test, y_test):
        Y_score = self.model.predict(X_test)
        Y_predict = self.model.predict_classes(X_test)
        Y_true = np.argmax(y_test, axis=1)

        return (Y_score, Y_predict, Y_true)

    def GetModel(self):
        return self.model

    def SetModel(self, model):
        self.model = model


    def SaveModel(self):
        model_json = self.model.to_json()
        with open("models/model.json", "w") as json_file:
            json_file.write(model_json)

        # Serialise weights
        self.model.save_weights("models/model.h5")

        print('Saved model to disk.')

class RNN_Model:
    pass

class LSTM_Model(Keras_Model):

    def __init__(self, nb_lstm_layers=0, nb_fc_layers=0, nb_units=0, output_dim=1, sequence_length=0, dropout=0.1, activation='relu', maskingvalue=None):
        Keras_Model.__init__(self)

        self.nb_lstm_layers = nb_lstm_layers
        self.nb_units = nb_units
        self.nb_fc_layers = nb_fc_layers
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

        for i in range(self.nb_lstm_layers):
            self.AddLSTMLayer()

        self.model.add(Flatten())

        self.AddOutputLayer()

    def AddMaskingLayer(self, value):
        # Masking layer
        self.model.add(Masking(mask_value=value, input_shape=(self.sequence_length, self.output_dim)))

    def AddInputLayer(self, return_seqs=True):
        # https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
        # Add masking layer is masking value exists
        if (self.cuddn is None):
            self.AddMaskingLayer(self.maskingvalue)
            self.model.add(LSTM(self.nb_units, return_sequences=return_seqs))
        else:
            self.model.add(CuDNNLSTM(self.nb_units, input_shape=(self.sequence_length, self.output_dim), return_sequences=return_seqs))

            # Batch Norm Layer
            self.model.add(BatchNormalization())

            self.AddDropoutLayer()

    def AddDropoutLayer(self):
        self.model.add(Dropout(self.dropout))

    def AddLSTMLayer(self):

        if self.cuddn:
            self.model.add(CuDNNLSTM(self.nb_units, return_sequences=True))
        else:
            self.model.add(LSTM(self.nb_units, return_sequences=True))

        # Batch Norm Layer
        self.model.add(BatchNormalization())

        self.AddDropoutLayer()



    def AddOutputLayer(self):

        self.model.add(Dense(self.nb_units))

        # Activation Layer
        if (self.activation == 'prelu'):
            self.model.add(PReLU())
        else:
            self.model.add(Activation(self.activation))

        self.model.add(Dense(int(self.nb_units/2)))

        # Activation Layer
        if (self.activation == 'prelu'):
            self.model.add(PReLU())
        else:
            self.model.add(Activation(self.activation))

        self.model.add(Dense(int(self.nb_units/4)))

        # Activation Layer
        if (self.activation == 'prelu'):
            self.model.add(PReLU())
        else:
            self.model.add(Activation(self.activation))

        self.model.add(Dense(self.output_dim))
        self.model.add(Activation('sigmoid'))

    def AddFCLayer(self, nb_units):
        self.model.add(Dense(nb_units))

        # Activation Layer
        if (self.activation == 'prelu'):
            self.model.add(PReLU())
        else:
            self.model.add(Activation(self.activation))



class CNN_Model(Keras_Model):

    def __init__(self, output_dim=1, sequence_length=10,
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

        print('Building CNN...')
        print('Filters: ', self.filters)
        print('Kernel Size: ', self.kernel_size)
        print('Number of blocks: ', self.nb_blocks)
        print('Pooling type:', self.pooling)
        print('Pooling Strides/Length:', self.pooling_val)
        print('Dropout: ', self.dropout)
        print('Activation:', self.activation)

        self.AddInputBlock()

        for i in range(self.nb_blocks):
            self.AddCNNBlock()

        self.AddOutputBlock()

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
        if (self.activation == 'prelu'):
            self.model.add(PReLU())
        else:
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
        if (self.activation == 'prelu'):
            self.model.add(PReLU())
        else:
            self.model.add(Activation(self.activation))

        # Dropout layer
        if (self.dropout is not None):
            self.model.add(Dropout(self.dropout))

    def AddOutputBlock(self):

        self.model.add(Flatten())

        self.model.add(Dense(64))

        # Activation Layer
        if (self.activation == 'prelu'):
            self.model.add(PReLU())
        else:
            self.model.add(Activation(self.activation))

        if self.dropout is not None:
            self.model.add(Dropout(self.dropout))

        self.model.add(Dense(32))

        # Activation Layer
        if (self.activation == 'prelu'):
            self.model.add(PReLU())
        else:
            self.model.add(Activation(self.activation))

        self.model.add(Dense(16))

        # Activation Layer
        if (self.activation == 'prelu'):
            self.model.add(PReLU())
        else:
            self.model.add(Activation(self.activation))

        self.model.add(Dense(8))

        # Activation Layer
        if (self.activation == 'prelu'):
            self.model.add(PReLU())
        else:
            self.model.add(Activation(self.activation))


        self.model.add(Dense(1, activation='sigmoid'))

    def SetSequenceLength(self, seq_length):
        self.sequence_length = seq_length

    def SetOutputDimension(self, dim):
        self.output_dim = dim