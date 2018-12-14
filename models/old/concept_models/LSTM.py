import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, classification_report, confusion_matrix

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Input, Masking, Dense, Activation, Dropout, LSTM
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping

# Load dataset

# https://machinelearningmastery.com/handle-long-sequences-long-short-term-memory-recurrent-neural-networks/
# https://machinelearningmastery.com/prepare-univariate-time-series-data-long-short-term-memory-networks/

train_data = pd.read_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//examples//kaggle_exoplanet_timeseries//exoTrain.csv')
test_data = pd.read_csv('C://Users//DYN//Google Drive//Intelligent_Systems_MSc//MSc_Project//data//examples//kaggle_exoplanet_timeseries//exoTest.csv')


# Split train and test sets
y_train = train_data.LABEL
y_test = test_data.LABEL
train_data = train_data.drop(labels='LABEL', axis=1)
test_data = test_data.drop(labels='LABEL', axis=1)
x_train = train_data.as_matrix().astype(np.float)
x_test = test_data.as_matrix().astype(np.float)

# Normalize/Scale/
train_scaler = MinMaxScaler(feature_range=(0,1))
test_scaler = MinMaxScaler(feature_range=(0,1))
train_scaler.fit(x_train)
test_scaler.fit(x_test)
x_train = train_scaler.transform(x_train)
x_test = test_scaler.transform(x_test)

# Reshape data to 3D input
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], 1)
#y_train = y_train.reshape(y_train.shape[0],y_train.shape[1])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], 1)

'''
Model parameters
'''

# Length of input timeseries
max_seq_length = 3197
# Number of output classes
nb_classes = 2
# Number of hidden units
nb_hidden = [16,16]


'''
Training parameters 
'''
batch_size = 32
nb_epochs = 1

print('Creating model...')
model = Sequential()

# https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/

# Masking layer
#model.add(Masking(mask_value=-1.))
model.add(LSTM(10, input_shape=(max_seq_length, 1), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(10))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

start = time.time()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Compilation Time: ", time.time() - start)
print(model.summary())


# Initialise plots

# Fit data
model.fit(x_train, y_train, batch_size, nb_epochs)

# Evaluate model
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)

Y_score = model.predict(x_test)
Y_predict = np.argmax(Y_score,axis=1)

print(Y_score)
print(Y_predict)
Y_true = np.argmax(y_test,axis=1)


#auc = roc_auc_score(y_test, Y_score, average='macro')

print('Classification Report: \n')
print(classification_report(y_test, Y_predict))

print('Confusion Matrix: \n')
conf_matrix = confusion_matrix(y_test, Y_predict)
print(conf_matrix)

conf_matrix_info = {
                1: {
                    'matrix': conf_matrix,
                    'title': 'Exoplanet Binary Classifier',
                   },
}

fix, ax = plt.subplots(figsize=(16, 12))
plt.suptitle('Confusion Matrix of Various Classifiers')
for ii, values in conf_matrix.items():
    matrix = values['matrix']
    title = values['title']
    plt.subplot(3, 3, ii) # starts from 1
    plt.title(title);
    sns.heatmap(matrix, annot=True,  fmt='');

sns.plt.show()


# prec = precision_score(y_test, Y_predict, average=None)
# rec = recall_score(y_test, Y_predict, average=None)
#
# print('Precision: ', prec)
# print('Recall: ', rec)

