#coding=utf-8

try:
    from data import *
except:
    pass

try:
    from model import *
except:
    pass

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform, conditional
except:
    pass

try:
    from keras.optimizers import Adam, SGD, RMSprop
except:
    pass

try:
    from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
except:
    pass

try:
    from evaluate import ModelEvaluator
except:
    pass

try:
    import json
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperas.distributions import conditional


#X, y = LoadOriginalData()
X, y = LoadDataset('lc_std_nanimputed.csv')
#X, y = LoadDataset('lc_std_nanmasked_SMOTE.csv')
#X, y = LoadDataset('lc_std.csv')

# Split data
#X_train, y_train, X_test, y_test = SplitData(X,y, test_size=0.2)
X_train, y_train, X_val, y_val, X_test, y_test = SplitData(X, y, test_size=0.2, val_set=True)

# Reshape data to 3D input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# y_train = y_train.reshape(y_train.shape[0], 1)
# y_test = y_test.reshape(y_test.shape[0], 1)

#return X_train, y_train, X_test, y_test


def keras_fmin_fnct(space):

#def create_cnn_model(X_train, y_train, X_test, y_test):

    cnn = CNN_Model(output_dim=1, sequence_length=X_train.shape[1],
                    nb_blocks=space['nb_blocks'], filters=space['filters'], kernel_size=space['kernel_size'],
                    activation=space['activation'], pooling=space['pooling'], pool_size=space['pool_size'], pool_strides=space['pool_size_1'],
                    dropout=space['dropout'])

    cnn.Build()

    opt = SGD(lr=0.1, momentum=0.25, decay=0.0001, nesterov=True)

    cnn.Compile(loss='binary_crossentropy', optimizer=SGD(lr=space['lr'], momentum=space['momentum'], decay=0.0001, nesterov=True), metrics=['accuracy'])

    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001, verbose=1)
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto')

    # hist = model.FitData(X_train, y_train, batch_size=batch_size, nb_epochs=nb_epochs, cb1=reduceLR)
    batch_size = space['batch_size']

    #cnn.FitData(X_train=X_train, y_train=y_train,
    #                         batch_size=batch_size, nb_epochs=50, verbose=2, cb1=reduceLR, cb2=earlyStopping)
    cnn.FitDataWithValidationCallbacks(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                                           batch_size=batch_size, nb_epochs=1, verbose=2, cb1=reduceLR, cb2=earlyStopping)

    score, acc = cnn.Evaluate(X_test, y_test, batch_size, verbose=2)

    print('Test Accuracy: ', acc)

    return {'loss': -acc, 'status': STATUS_OK, 'model': cnn.GetModel(), 'X_test': X_test, 'y_test': y_test}

def get_space():
    return {
        'nb_blocks': hp.choice('nb_blocks', [1,2,3]),
        'filters': hp.choice('filters', [8, 16, 32, 64]),
        'kernel_size': hp.choice('kernel_size', [5, 8, 11]),
        'activation': hp.choice('activation', ['prelu']),
        'pooling': hp.choice('pooling', ['max', 'average']),
        'pool_size': hp.choice('pool_size', [1,2,3,4,5]),
        'pool_size_1': hp.choice('pool_size_1', [1,2,3,4,5]),
        'dropout': hp.uniform('dropout', 0,1),
        'lr': hp.choice('lr', [0.1,0.01,0.001]),
        'momentum': hp.choice('momentum', [0., 0.25]),
        'batch_size': hp.choice('batch_size', [16,32]),
    }
