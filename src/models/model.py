import os

from definitions import MODELS_OUTPUT_DIR
from keras.api.models import Sequential
from keras.api.layers import LSTM, Conv1D, MaxPooling1D, AveragePooling1D, Dropout, Dense, Flatten, BatchNormalization


class CNNModel:
    def __init__(self, filters, kernel_size, pooling, pooling_size, pooling_strides, conv_dropout, fc_dropout, fc_units,
                 nb_blocks, batch_norm, activation='relu'):
        self.filters = filters
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.pooling_size = pooling_size
        self.pooling_strides = pooling_strides
        self.conv_dropout = conv_dropout
        self.fc_dropout = fc_dropout
        self.fc_units = fc_units
        self.nb_blocks = nb_blocks
        self.batch_norm = batch_norm
        self.activation = activation

        self.name = self._generate_name()

        self.model = self.build_model()

    def _generate_name(self):
        pooling_abbr = 'MP' if self.pooling == 'max' else 'AP'  # MaxPooling (MP) or AveragePooling (AP)
        bn = 'BN' if self.batch_norm else 'NoBN'
        return f"CNN_{self.filters}F_{self.kernel_size}K_{pooling_abbr}_{bn}_{self.activation}"

    def build_model(self):
        model = Sequential(name=self.name)

        for _ in range(self.nb_blocks):
            model.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation=self.activation))
            if self.pooling == 'max':
                model.add(MaxPooling1D(pool_size=self.pooling_size, strides=self.pooling_strides))
            elif self.pooling == 'average':
                model.add(AveragePooling1D(pool_size=self.pooling_size, strides=self.pooling_strides))
            model.add(Dropout(self.conv_dropout))

            if self.batch_norm:
                model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(self.fc_units, activation=self.activation))
        model.add(Dropout(self.fc_dropout))
        model.add(Dense(1, activation='sigmoid'))  # Binary classification

        return model

    def fit(self, X_train, y_train, batch_size, epochs, validation_data=None, verbose=1):
        return self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=validation_data,
                              verbose=verbose)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y, batch_size=32, verbose=1):
        return self.model.evaluate(X, y, batch_size=batch_size, verbose=verbose)

    def save(self, filepath):
        self.model.save(filepath)


class LSTMModel:
    def __init__(self, nb_lstm_layers, lstm_units, dropout, fc_units, fc_layers, batch_norm, activation='relu'):
        self.nb_lstm_layers = nb_lstm_layers
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.fc_units = fc_units
        self.fc_layers = fc_layers
        self.batch_norm = batch_norm
        self.activation = activation
        self.name = self._generate_name()
        self.model = self.build_model()

    def _generate_name(self):
        # Generate a name based on key parameters to keep it short but descriptive
        bn = 'BN' if self.batch_norm else 'NoBN'
        return f"LSTM_{self.nb_lstm_layers}L_{self.lstm_units}U_{bn}_{self.activation}"

    def build_model(self):
        model = Sequential(name=self.name)

        for _ in range(self.nb_lstm_layers):
            model.add(LSTM(self.lstm_units, return_sequences=True))
            model.add(Dropout(self.dropout))
            if self.batch_norm:
                model.add(BatchNormalization())

        model.add(Flatten())

        for _ in range(self.fc_layers):
            model.add(Dense(self.fc_units, activation=self.activation))
            model.add(Dropout(self.dropout))

        model.add(Dense(1, activation='sigmoid'))  # Binary classification

        return model

    def fit(self, X_train, y_train, batch_size, epochs, validation_data=None, verbose=1):
        return self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=validation_data,
                              verbose=verbose)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y, batch_size=32, verbose=1):
        return self.model.evaluate(X, y, batch_size=batch_size, verbose=verbose)

    def save(self, filepath):
        self.model.save(filepath)
