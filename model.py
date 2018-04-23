import json
import numpy as np
import time
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Masking, Dense, Activation, Dropout,CuDNNLSTM, LSTM, Conv1D, Flatten, GlobalAveragePooling1D, BatchNormalization, MaxPool1D, AveragePooling1D, PReLU
from keras.optimizers import Adam, SGD
from sklearn.preprocessing import MinMaxScaler


class Keras_Model:
    def __init__(self):

        self.model = Sequential()

    def Compile(self, loss, optimizer, metrics='accuracy'):
        start = time.time()
        self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        print("Compilation Time: ", time.time() - start)
        print(self.model.summary())


    def FitData(self, X_train, y_train, batch_size, nb_epochs, verbose=1, validation_split = None):

        if validation_split is not None :
            hist = self.model.fit(X_train, y_train, batch_size, nb_epochs, validation_split=validation_split, verbose=verbose)
        else:
            hist = self.model.fit(X_train, y_train, batch_size, nb_epochs, verbose=verbose)

        return hist

    def FitDataWithValidation(self, X_train, y_train, X_val, y_val, batch_size, nb_epochs, verbose = 1):
        return self.model.fit(X_train, y_train, batch_size, nb_epochs, verbose=verbose, validation_data=(X_val, y_val))

    def FitDataWithValidationCallbacks(self, X_train, y_train, X_val, y_val, batch_size, nb_epochs, cb1, verbose):
        return self.model.fit(X_train, y_train, batch_size, nb_epochs, verbose=verbose, validation_data=(X_val, y_val), callbacks=[cb1])

    def Evaluate(self, X_test, y_test, batch_size, verbose=1):

        score, acc = self.model.evaluate(X_test, y_test, batch_size, verbose=verbose)
        return score, acc

    def Predict(self, X_test, y_test, segment=False):

        #if (X_test.shape[1] != self.sequencelength):

        if (segment):

            npts = 180
            stepsize = 2
            scaler = MinMaxScaler()

            for i in range(X_test.shape[0]):

                Y_score = np.array(X_test.shape[0])

                PTS = np.array(np.zeros(X_test.shape[1]))

                for k in range(X_test.shape[1] - npts):

                    predictionSegment = X_test[i,k:k+npts]
                    predictionSegment = predictionSegment.reshape(predictionSegment.shape[1], predictionSegment.shape[0], 1)
                    prob = self.model.predict(predictionSegment)

                    PTS[k:k + npts] = PTS[k:k + npts] + prob
                    k += stepsize

                normalizedPTS = scaler.fit_transform(PTS.reshape(-1,1))
                #normalizedPTS = (1 - 0) / ( np.sum(PTS) - 0 ) * (np.sum(PTS) - np.sum(PTS) ) + 1

                normalizedPTS = np.mean(normalizedPTS)

                Y_score = np.append(Y_score, normalizedPTS)

                print('Evaluated {} out of {} samples.'.format(i, X_test.shape[0]))

            if Y_score.shape[-1] > 1:
                Y_predict = Y_score.argmax(axis=-1)
            else:
                Y_predict = (Y_score > 0.5).astype('int32')
        else:

            Y_score = self.model.predict(X_test)
            Y_predict = self.model.predict_classes(X_test)
            Y_true = np.argmax(y_test, axis=1)

        return (Y_score, Y_predict, Y_true)

    def GetModel(self):
        return self.model

    def SetModel(self, model):
        self.model = model


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
                 conv_dropout=0.2, fc_dropout = 0.5, dense_units = 64, batch_norm = True):

        print('Initialising CNN Model... \n')

        Keras_Model.__init__(self)


        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation

        self.pooling = pooling
        self.conv_dropout = conv_dropout
        self.fc_dropout = fc_dropout
        self.dense_units = dense_units
        self.batch_norm = batch_norm

        if self.pooling == 'max':
            self.pooling_val = pool_strides
        elif self.pooling == 'average':
            self.pooling_val = pool_size

    def Build(self):

        print('Building CNN... \n' )
        print('Filters: ', self.filters)
        print('Kernel Size: ', self.kernel_size)
        print('Number of blocks: ', self.nb_blocks)
        print('Pooling type:', self.pooling)
        print('Pooling Strides/Length:', self.pooling_val)
        print('Conv Dropout: ', self.conv_dropout)
        print('FC Dropout: ', self.fc_dropout)
        print('Dense Units: ', self.dense_units)
        print('Activation:', self.activation)
        print('Batch Normalisation:', self.batch_norm)
        print('\n')

        self.AddInputBlock()

        for i in range(self.nb_blocks):
            self.AddCNNBlock()

        self.AddOutputBlock()

    def AddInputBlock(self):

        self.model.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size, input_shape=(self.sequence_length, self.output_dim)))

        # Batch Norm Layer
        if (self.batch_norm):
            self.model.add(BatchNormalization())

        # Pooling Layer
        if (self.pooling == 'max'):
            self.model.add(MaxPool1D(strides=self.pooling_val))
        elif (self.pooling == 'average'):
            self.model.add(AveragePooling1D(pool_size=self.pooling_val))

        # Dropout layer
        if (self.conv_dropout is not None):
            self.model.add(Dropout(self.conv_dropout))

        # Activation Layer
        if (self.activation == 'prelu'):
            self.model.add(PReLU())
        else:
            self.model.add(Activation(self.activation))


    def AddCNNBlock(self):

        # Convolution Layer
        self.model.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size))

        # Batch Norm Layer
        if (self.batch_norm):
            self.model.add(BatchNormalization())


        # Pooling Layer
        if (self.pooling == 'max'):
            self.model.add(MaxPool1D(strides=self.pooling_val))
        elif (self.pooling == 'average'):
            self.model.add(AveragePooling1D(pool_size=self.pooling_val))

        # Dropout layer
        if (self.conv_dropout is not None):
            self.model.add(Dropout(self.conv_dropout))

        # Activation Layer
        if (self.activation == 'prelu'):
            self.model.add(PReLU())
        else:
            self.model.add(Activation(self.activation))


    def AddOutputBlock(self):

        self.model.add(Flatten())

        self.model.add(Dense(self.dense_units))

        # Dropout layer
        if self.fc_dropout is not None:
            self.model.add(Dropout(self.fc_dropout))

        # Activation Layer
        if (self.activation == 'prelu'):
            self.model.add(PReLU())
        else:
            self.model.add(Activation(self.activation))


        self.model.add(Dense(int(self.dense_units)))

        # Activation Layer
        if (self.activation == 'prelu'):
            self.model.add(PReLU())
        else:
            self.model.add(Activation(self.activation))

        if self.fc_dropout is not None:
            self.model.add(Dropout(self.fc_dropout))

        self.model.add(Dense(int(self.dense_units)))

        # Activation Layer
        if (self.activation == 'prelu'):
            self.model.add(PReLU())
        else:
            self.model.add(Activation(self.activation))

        if self.fc_dropout is not None:
            self.model.add(Dropout(self.fc_dropout))

        self.model.add(Dense(int(self.dense_units)))

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

    def SaveModel(self, name, weights=False):

        root = 'models/'

        # Save model with weights and training configuration
        if (weights):
            model_json = self.model.to_json()
            with open(root + name + '.json', "w") as json_file:
                json_file.write(model_json)

            # Serialise weights
            self.model.save_weights(root + name + '.h5')

        # Save configuration for rebuilding model
        else:
            dict = {}

            dict['nb_blocks'] = self.nb_blocks
            dict['filters'] = self.filters

            dict['kernel_size'] = self.kernel_size
            dict['activation'] = self.activation

            if self.pooling == 'max':
                dict['pooling_strides'] = self.pooling_val
            elif self.pooling == 'average':
                dict['pooling_size'] = self.pooling_val

            dict['conv_dropout'] = self.conv_dropout
            dict['fc_dropout'] = self.fc_dropout
            dict['fc_units'] = self.dense_units
            dict['batch_norm'] = self.batch_norm

            with open(root + 'configs/' + name + '.json', 'w') as fp:
                json.dump(dict, fp)



        print('Saved model to disk.')

    def LoadConfiguration(self, config):

        self.nb_blocks = config['nb_blocks']
        self.filters = config['filters']
        self.kernel_size = config['kernel_size']
        self.activation = config['activation']

        self.pooling = config['pooling']
        if self.pooling == 'max':
            self.pooling_val = config['pooling_strides']
        elif self.pooling == 'average':
            self.pooling_val = config['pooling_size']

        self.conv_dropout = config['conv_dropout']
        self.fc_dropout = config['fc_dropout']
        self.dense_units = config['fc_units']
        self.batch_norm = config['batch_norm']


        self.Build()
        self.Compile(loss='binary_crossentropy',
                    optimizer=SGD(lr=0.001 * config['lr_rate_mult'], momentum=config['momentum'], decay=0.0001,
                                  nesterov=True), metrics=['accuracy'])

        print('Loaded and compiled Keras model succesfully. \n')
