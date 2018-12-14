import json
import numpy as np
import time
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Masking, Dense, Activation, Dropout,CuDNNLSTM, LSTM, Conv1D, Flatten, GlobalAveragePooling1D, BatchNormalization, MaxPool1D, AveragePooling1D, PReLU
from keras.optimizers import Adam, SGD
from sklearn.preprocessing import MinMaxScaler


class Keras_Model:
    """
    A Keras Sequential Model Wrapper
    """
    def __init__(self):
        """
        Keras Model Constructor
        """

        self.model = Sequential()

    def Compile(self, loss, optimizer, metrics='accuracy'):
        """
        Compile the Keras Sequential Model

        Args:
            loss: Loss function to use
            optimizer: Optimizer to use
            metrics: Metric to validate performance
        """
        start = time.time()
        self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        print("Compilation Time: ", time.time() - start)
        print(self.model.summary())


    def FitData(self, X_train, y_train, batch_size, nb_epochs, verbose=1, validation_split = None):
        """
        Fits training data to the model

        Args:
            X_train: Features to fit during training
            y_train: Target column to fit during training
            batch_size: Number of batches at a time
            nb_epochs: Number of epochs to train the model
            verbose: Log verbosity
            validation_split: Take a split from the training data for validation?

        Returns: Model History containing training performance

        """

        if validation_split is not None :
            hist = self.model.fit(X_train, y_train, batch_size, nb_epochs, validation_split=validation_split, verbose=verbose)
        else:
            hist = self.model.fit(X_train, y_train, batch_size, nb_epochs, verbose=verbose)

        return hist

    def FitDataWithValidation(self, X_train, y_train, X_val, y_val, batch_size, nb_epochs, verbose = 1):
        """
        Fits training data to the model but uses seperate validation data

        Args:
            X_train: Features to fit during training
            y_train: Target column to fit during training
            X_val: Features to fit during validation
            y_val: Target column to fit during validation
            batch_size: Number of batches at a time
            nb_epochs: Number of epochs to train the model
            verbose: Log verbosity

        Returns: Model History containing training performance

        """
        return self.model.fit(X_train, y_train, batch_size, nb_epochs, verbose=verbose, validation_data=(X_val, y_val))

    def FitDataWithValidationCallbacks(self, X_train, y_train, X_val, y_val, batch_size, nb_epochs, cb1, cb2, verbose):
        """
        Fits training data to the model but uses seperate validation data and callbacks

        Args:
            X_train: Features to fit during training
            y_train: Target column to fit during training
            X_val: Features to fit during validation
            y_val: Target column to fit during validation
            batch_size: Number of batches at a time
            nb_epochs: Number of epochs to train the model
            cb1: Callback 1
            cb2: Callback 2
            verbose: Log verbosity

        Returns: Model History containing training performance

        """
        return self.model.fit(X_train, y_train, batch_size, nb_epochs, verbose=verbose, validation_data=(X_val, y_val), callbacks=[cb1, cb2])

    def Evaluate(self, X_test, y_test, batch_size, verbose=1):
        """
        Evaluates the fitted model on new data

        Args:
            X_test: Features to fit during training
            y_test: Target column to fit during training
            batch_size: Number of batches at a time
            verbose: Log verbosity

        Returns: Proba

        """

        score, acc = self.model.evaluate(X_test, y_test, batch_size, verbose=verbose)
        return score, acc

    def Predict(self, X_test, y_test):
        """

        Args:
            X_test: Features to predict
            y_test: Target column to validate predictions

        Returns:
            Y_score = Probabilies for each sample in the test data
            Y_predict = Predicted classes for each sample in the test data
            Y_true = True labels

        """

        Y_score = self.model.predict(X_test)
        Y_predict = self.model.predict_classes(X_test)
        Y_true = np.argmax(y_test, axis=1)

        return (Y_score, Y_predict, Y_true)

    def GetModel(self):
        """

        Returns: Keras sequential model

        """
        return self.model

    def SetModel(self, model):
        """
        Overwrites the current Keras Sequential model
        Args:
            model: Keras sequential model
        """
        self.model = model


class LSTM_Model(Keras_Model):
    """
    LSTM child of KerasModel
    """

    def __init__(self, nb_lstm_layers=0, nb_lstm_units=10, nb_fc_layers=0, nb_fc_units=32, output_dim=1, sequence_length=0, dropout=0.1, activation='relu', batch_normalisation=True):
        """
        LSTM Constructor

        Args:
            nb_lstm_layers: Number of LSTM layers
            nb_lstm_units: Number of LSTM hidden units
            nb_fc_layers: Number of Fully Connected Layers
            nb_fc_units: Number of Hidden Units in the Fully Connected Layers
            output_dim: Output dimensionality
            sequence_length: Length of the input sequence
            dropout: Dropout fraction
            activation: Activation function to use
            batch_normalisation: Boolean flag for batch normalisation layers
        """
        Keras_Model.__init__(self)

        print('Initialising LSTM Model... \n')

        self.nb_lstm_layers = nb_lstm_layers
        self.nb_lstm_units = nb_lstm_units
        self.nb_fc_layers = nb_fc_layers
        self.nb_fc_units = nb_fc_units
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_normalisation

    def Build(self):
        """
        Builds the topology of the network
        """

        print('Building LSTM... \n' )
        print('Number of LSTM layers: ', self.nb_lstm_layers)
        print('Number of LSTM units: ', self.nb_lstm_units)
        print('Number of FC layers: ', self.nb_fc_layers)
        print('Number of FC units: ', self.nb_fc_units)
        print('Dropout: ', self.dropout)
        print('Activation:', self.activation)
        print('Batch Normalisation:', self.batch_norm)

        print('\n')

        self.AddInputLayer()

        for i in range(self.nb_lstm_layers):
            self.AddLSTMLayer()

        self.model.add(Flatten())

        self.AddOutputLayer()

    def AddMaskingLayer(self, value):
        """

        Args:
            value:
        """
        # Masking layer
        self.model.add(Masking(mask_value=value, input_shape=(self.sequence_length, self.output_dim)))

    def AddInputLayer(self, return_seqs=True):
        """

        Args:
            return_seqs:
        """

        # Add masking layer is masking value exists
        self.model.add(CuDNNLSTM(self.nb_lstm_units, input_shape=(self.sequence_length, self.output_dim), return_sequences=return_seqs))

        # Batch Norm Layer
        self.model.add(BatchNormalization())

        self.AddDropoutLayer()

        # Activation Layer
        if (self.activation == 'prelu'):
            self.model.add(PReLU())
        else:
            self.model.add(Activation(self.activation))


    def AddDropoutLayer(self):
        """

        """
        self.model.add(Dropout(self.dropout))

    def AddLSTMLayer(self):
        """

        """

        self.model.add(CuDNNLSTM(self.nb_lstm_units, return_sequences=True))

        # Batch Norm Layer
        if (self.batch_norm):
            self.model.add(BatchNormalization())

        self.AddDropoutLayer()

        # Activation Layer
        if (self.activation == 'prelu'):
            self.model.add(PReLU())
        else:
            self.model.add(Activation(self.activation))



    def AddOutputLayer(self):
        """

        """


        for i in range(self.nb_fc_layers):
            self.AddFCLayer()

        self.model.add(Dense(self.output_dim))
        self.model.add(Activation('sigmoid'))


    def AddFCLayer(self):
        """

        """

        self.model.add(Dense(self.nb_fc_units))

        # Dropout layer
        self.model.add(Dropout(self.dropout))

        # Activation Layer
        if (self.activation == 'prelu'):
            self.model.add(PReLU())
        else:
            self.model.add(Activation(self.activation))

    def LoadLSTMConfiguration(self, config):
        """

        Args:
            config:
        """

        self.nb_lstm_layers = config['lstm_layers']
        self.nb_lstm_units = config['lstm_units']
        self.nb_fc_layers = config['fc_layers']
        self.nb_fc_units = config['fc_units']
        self.dropout = config['dropout']
        self.activation = config['activation']
        self.batch_norm = config['batch_norm']


        self.Build()
        self.Compile(loss='binary_crossentropy',
                             optimizer=SGD(lr=0.001 * config['lr_rate_mult'], momentum=config['momentum'], decay=0.0001,
                                           nesterov=True), metrics=['accuracy'])

        print('Loaded and compiled Keras model succesfully. \n')

    def SaveLSTMModel(self, name, config=False, weights=False):
        """

        Args:
            name:
            config:
            weights:
        """

        root = 'models/'

        # Save model with weights and training configuration
        if (weights):
            model_json = self.model.to_json()
            with open(root + name + '.json', "w") as json_file:
                json_file.write(model_json)

            # Serialise weights
            self.model.save_weights(root + name + '.h5')

            if(config):
                with open(root + 'configs/' + name + '.json', 'w') as fp:
                    json.dump(config, fp)

        # Save configuration for rebuilding model
        else:
            dict = {}

            dict['lstm_layers']= self.nb_lstm_layers
            dict['lstm_units']= self.nb_lstm_units
            dict['fc_layers']= self.nb_fc_layers
            dict['fc_units']= self.nb_fc_units
            dict['dropout']= self.dropout
            dict['activation']= self.activation
            dict['batch_norm']= self.batch_norm
            with open(root + 'configs/' + name + '.json', 'w') as fp:
                json.dump(dict, fp)


        print('Saved model to disk.')







class CNN_Model(Keras_Model):
    """
    """

    def __init__(self, output_dim=1, sequence_length=10,
                 nb_blocks=1,filters=8, kernel_size=5, activation = 'relu',
                 pooling='max', pool_strides = 2, pool_size = 5,
                 conv_dropout=0.2, fc_dropout = 0.5, dense_units = 64, batch_norm = True):
        """

        Args:
            output_dim: Output Dimensionality
            sequence_length: Length of the input sequences
            nb_blocks: Number of convolutional blocks
            filters: Number of filters in the convolutional layers
            kernel_size: Size of the kernel to use in convolutional layers
            activation: Activation function
            pooling: Type of pooling (Max or Average)
            pool_strides: Pooling strides
            pool_size: Pooling window size
            conv_dropout: Convolutional layer dropout
            fc_dropout: Fully Connected layer dropout
            dense_units: Number of hidden units in fully connected layers
            batch_norm: Boolean flag specifying whether to use batch normalisation layers
        """

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

        self.pooling_strides = pool_strides
        self.pooling_size = pool_size

    def Build(self):
        """
        Builds the topology of the network
        """

        print('Building CNN... \n' )
        print('Filters: ', self.filters)
        print('Kernel Size: ', self.kernel_size)
        print('Number of blocks: ', self.nb_blocks)
        print('Pooling type:', self.pooling)
        print('Pooling Strides:', self.pooling_strides)
        print('Pooling Length:', self.pooling_size)

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
        """

        """

        self.model.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size, input_shape=(self.sequence_length, self.output_dim)))

        # Batch Norm Layer
        if (self.batch_norm):
            self.model.add(BatchNormalization())

        # Pooling Layer
        if (self.pooling == 'max'):
            self.model.add(MaxPool1D(pool_size=self.pooling_size,strides=self.pooling_strides))
        elif (self.pooling == 'average'):
            self.model.add(AveragePooling1D(pool_size=self.pooling_size, strides=self.pooling_strides))

        # Dropout layer
        if (self.conv_dropout is not None):
            self.model.add(Dropout(self.conv_dropout))

        # Activation Layer
        if (self.activation == 'prelu'):
            self.model.add(PReLU())
        else:
            self.model.add(Activation(self.activation))


    def AddCNNBlock(self):
        """

        """

        # Convolution Layer
        self.model.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size))


        # Batch Norm Layer
        if (self.batch_norm):
            self.model.add(BatchNormalization())


        # Pooling Layer
        if (self.pooling == 'max'):
            self.model.add(MaxPool1D(pool_size=self.pooling_size,strides=self.pooling_strides))
        elif (self.pooling == 'average'):
            self.model.add(AveragePooling1D(pool_size=self.pooling_size, strides=self.pooling_strides))

        # Dropout layer
        if (self.conv_dropout is not None):
            self.model.add(Dropout(self.conv_dropout))

        # Activation Layer
        if (self.activation == 'prelu'):
            self.model.add(PReLU())
        else:
            self.model.add(Activation(self.activation))


    def AddOutputBlock(self):
        """

        """

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
        """

        Args:
            seq_length:
        """
        self.sequence_length = seq_length

    def SetOutputDimension(self, dim):
        """

        Args:
            dim:
        """
        self.output_dim = dim

    def SaveCNNModel(self, name, config=False, weights=False):
        """

        Args:
            name:
            config:
            weights:
        """

        root = 'models/'

        # Save model with weights and training configuration
        if (weights):
            model_json = self.model.to_json()
            with open(root + name + '.json', "w") as json_file:
                json_file.write(model_json)

            # Serialise weights
            self.model.save_weights(root + name + '.h5')

            if(config):
                with open(root + 'configs/' + name + '.json', 'w') as fp:
                    json.dump(config, fp)

        # Save configuration for rebuilding model
        else:
            dict = {}

            dict['nb_blocks'] = self.nb_blocks
            dict['filters'] = self.filters

            dict['kernel_size'] = self.kernel_size
            dict['activation'] = self.activation

            dict['pooling'] = self.pooling
            dict['pooling_strides'] = self.pooling_strides
            dict['pooling_size'] = self.pooling_size

            dict['conv_dropout'] = self.conv_dropout
            dict['fc_dropout'] = self.fc_dropout
            dict['fc_units'] = self.dense_units
            dict['batch_norm'] = self.batch_norm

            with open(root + 'configs/' + name + '.json', 'w') as fp:
                json.dump(dict, fp)



        print('Saved model to disk.')

    def LoadCNNConfiguration(self, config):
        """

        Args:
            config:
        """

        self.nb_blocks = config['nb_blocks']
        self.filters = config['filters']
        self.kernel_size = config['kernel_size']
        self.activation = config['activation']

        self.pooling = config['pooling']
        self.pooling_strides = config['pooling_strides']
        self.pooling_size = config['pooling_size']

        self.conv_dropout = config['conv_dropout']
        self.fc_dropout = config['fc_dropout']
        self.dense_units = config['fc_units']
        self.batch_norm = config['batch_norm']


        self.Build()
        self.Compile(loss='binary_crossentropy',
                    optimizer=SGD(lr=0.001 * config['lr_rate_mult'], momentum=config['momentum'], decay=0.0001,
                                  nesterov=True), metrics=['accuracy'])

        print('Loaded and compiled Keras model succesfully. \n')

    def LoadConfigurationFromFile(self, config_name):
        """

        Args:
            config_name:
        """

        root = 'models/configs/'

        # Load json configuration
        json_file = open(root + config_name + '.json', 'r')
        config_str = json_file.read()
        config = json.loads(config_str)
        json_file.close()

        self.nb_blocks = config['nb_blocks']
        self.filters = config['filters']
        self.kernel_size = config['kernel_size']
        self.activation = config['activation']

        self.pooling = config['pooling']
        self.pooling_strides = config['pooling_strides']
        self.pooling_size = config['pooling_size']

        self.conv_dropout = config['conv_dropout']
        self.fc_dropout = config['fc_dropout']
        self.dense_units = config['fc_units']
        self.batch_norm = config['batch_norm']

        #self.Build()
        #self.Compile(loss='binary_crossentropy',
        #             optimizer=SGD(lr=0.001 * config['lr_rate_mult'], momentum=config['momentum'], decay=0.0001,
        #                           nesterov=True), metrics=['accuracy'])

        #print('Loaded and compiled Keras model succesfully. \n')

