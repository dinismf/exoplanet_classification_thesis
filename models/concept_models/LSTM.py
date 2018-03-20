import time
import pandas as pd

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Embedding, Dense, Dropout, LSTM

from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping


# Load dataset

# https://machinelearningmastery.com/handle-long-sequences-long-short-term-memory-recurrent-neural-networks/
# https://machinelearningmastery.com/prepare-univariate-time-series-data-long-short-term-memory-networks/

train_data = pd.read_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//examples//kaggle_exoplanet_timeseries//exoTrain.csv')
test_data = pd.read_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//examples//kaggle_exoplanet_timeseries//exoTest.csv')

train_labels = train_data.LABEL
test_labels = test_data.LABEL
train_data = train_data.drop(labels='LABEL', axis=1)
test_data = test_data.drop(labels='LABEL', axis=1)

np_train = train_data.as_matrix()

# Length of input timeseries
max_seq_length = 4500

print('Creating model...')
model = Sequential()

# https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
model.add(LSTM(32, input_shape=(max_seq_length, 1)))
model.add(Dropout(0.2))

#model.add(LSTM(100))
#model.add(Dropout(0.2))


model.add(Dense(1, activation='sigmoid'))


start = time.time()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Compilation Time: ", time.time() - start)

print(model.summary())

# Fit data

# Evaluate model