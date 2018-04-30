import datetime
from data import *
from model import *
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
from hyperas.utils import eval_hyperopt_space
from keras.optimizers import Adam, SGD, RMSprop
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from evaluate import ModelEvaluator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, precision_score
import json

# def data():
#
#     #X, y = LoadOriginalData()
#     X, y = LoadDataset('lc_std_nanimputed.csv')
#     #X, y = LoadDataset('lc_std_nanmasked_SMOTE.csv')
#     #X, y = LoadDataset('lc_std.csv')
#
#     # Split data
#     #X_train, y_train, X_test, y_test = SplitData(X,y, test_size=0.2)
#     X_train, y_train, X_val, y_val, X_test, y_test = SplitData(X, y, test_size=0.2, val_set=True)
#
#     # Reshape data to 3D input
#     X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
#     X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
#     X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
#
#     # y_train = y_train.reshape(y_train.shape[0], 1)
#     # y_test = y_test.reshape(y_test.shape[0], 1)
#
#     #return X_train, y_train, X_test, y_test
#     return X_train, y_train, X_val, y_val, X_test, y_test

X, y = LoadDataset('binned_confirmed_fps_binned.csv')
# X, y = LoadDataset('lc_std_nanmasked_SMOTE.csv')
# X, y = LoadDataset('lc_std.csv')

# Split data
X_train, y_train, X_test, y_test = SplitData(X,y, test_size=0.2)
#X_train, y_train, X_val, y_val, X_test, y_test = SplitData(X, y, test_size=0.2, val_set=True)


#Remove any NaNs
X_train = MissingValuesHandler(X_train).imputeNaN()
X_test = MissingValuesHandler(X_test).imputeNaN()

# Standardize training data
X_train = Standardizer().standardize(X_train, na_values=False)
X_test = Standardizer().standardize(X_test, na_values=False)


# Save the split train and test dataset before optimization for future reference
y_train_save = pd.DataFrame(y_train, columns=['LABEL'])
X_train_save = pd.DataFrame(X_train.astype(np.float))
y_test_save = pd.DataFrame(y_test, columns=['LABEL'])
X_test_save = pd.DataFrame(X_test.astype(np.float))

X_train_save.to_csv('data//testing_data//1st_binneddata_XTRAIN.csv', index=False)
y_train_save.to_csv('data//testing_data//1st_binneddata_YTRAIN.csv', index=False)
X_test_save.to_csv('data//testing_data//1st_binneddata_XTEST.csv', index=False)
y_test_save.to_csv('data//testing_data//1st_binneddata_YTEST.csv', index=False)



#print('Reshaping the input data to 3D')

# Reshape data to 3D input
#X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
#X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# space = {
#
#         'nb_blocks': hp.choice('nb_blocks', [0,1,2,3]),
#         'filters': hp.choice('filters', [8,16,32,64]),
#         'kernel_size':  hp.choice('kernel_size', [3, 5, 7, 9]),
#         'pooling': hp.choice('pooling', ['max','average']),
#         'pooling_size': hp.choice('pooling_size', [2,3,4]),
#         'pooling_strides': hp.choice('pooling_strides', [2,3,4]),
#         'conv_dropout': hp.uniform('conv_dropout', 0.0, 0.35),
#         'fc_dropout': hp.uniform('fc_dropout', 0.0,0.6),
#         'fc_units': hp.choice('fc_units', [32,64,128]),
#         'batch_size' : hp.choice('batch_size', [16,32]),
#         'lr_rate_mult': hp.loguniform('lr_rate_mult', -0.5, 0.5),
#         'momentum': hp.choice('momentum', [0, 0.25, 0.4]),
#         'batch_norm': hp.choice('batch_norm', [True, False]),
#
#         #'nb_epochs' :  35,
#         'nb_epochs' :  hp.uniform('nb_epochs', 5.0, 40.0),
#         'activation': 'prelu'
#         }

space = {

        'nb_blocks': hp.choice('nb_blocks', [0,1,2,3]),
        'filters': hp.choice('filters', [8,16,32,64,128,254]),
        'kernel_size':  hp.choice('kernel_size', [3, 5, 7, 9]),
        'pooling': hp.choice('pooling', ['max','average']),
        'pooling_size': hp.choice('pooling_size', [2,3,4]),
        'pooling_strides': hp.choice('pooling_strides', [2,3,4]),
        'conv_dropout': hp.uniform('conv_dropout', 0.0, 0.35),
        'fc_dropout': hp.uniform('fc_dropout', 0.0,0.6),
        'fc_units': hp.choice('fc_units', [32,64,128,254]),
        'batch_size' : hp.choice('batch_size', [16,32]),
        'lr_rate_mult': hp.loguniform('lr_rate_mult', -0.5, 0.5),
        'momentum': hp.choice('momentum', [0, 0.25, 0.4]),
        'batch_norm': hp.choice('batch_norm', [True, False]),

        'nb_epochs' :  hp.uniform('nb_epochs', 5.0, 50.0),
        'activation': 'prelu'
        }


def create_cnn_model(params):

    start_time = time.time()

    cnn = CNN_Model(output_dim=1, sequence_length=X_train.shape[1], nb_blocks=params['nb_blocks'], filters=params['filters'],
                    kernel_size=params['kernel_size'],
                    activation=params['activation'], pooling=params['pooling'], pool_size=params['pooling_size'],
                    pool_strides=params['pooling_strides'],
                    conv_dropout=params['conv_dropout'], fc_dropout=params['fc_dropout'], dense_units=params['fc_units'], batch_norm=params['batch_norm'])

    cnn.Build()

    cnn.Compile(loss='binary_crossentropy', optimizer=SGD(lr= 0.001 * params['lr_rate_mult'], momentum=params['momentum'], decay=0.0001,
                          nesterov=True), metrics=['accuracy'])

    nb_epochs = int(params['nb_epochs'])
    print('Number of Epochs: ', nb_epochs)
    batch_size = params['batch_size']

    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

    cvscores = []

    for train_index, valid_index in kfold.split(X_train, y_train):

        X_train_fold = X_train[train_index]
        X_valid_fold = X_train[valid_index]
        y_train_fold = y_train.iloc[train_index]
        y_valid_fold = y_train.iloc[valid_index]

        # Reshape data to 3D input
        X_train_fold = X_train_fold.reshape(X_train_fold.shape[0], X_train_fold.shape[1], 1)
        # X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        X_valid_fold = X_valid_fold.reshape(X_valid_fold.shape[0], X_valid_fold.shape[1], 1)

        #reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001, verbose=1)
        #earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto')
        cnn.FitData(X_train=X_train_fold, y_train=y_train_fold, batch_size = batch_size, nb_epochs = nb_epochs, verbose = 2)
        #cnn.FitDataWithValidationCallbacks(X_train=X[train], y_train=y[train], X_val=X[valid], y_val=y[valid],
        #                                   batch_size=batch_size, nb_epochs=50, verbose=2, cb1=earlyStopping)

        score, acc = cnn.Evaluate(X_valid_fold, y_valid_fold, batch_size, verbose=0)

        Y_score, Y_predict, Y_true = cnn.Predict(X_valid_fold, y_valid_fold)
        recall = recall_score(y_valid_fold, Y_predict)
        precision = precision_score(y_valid_fold, Y_predict)
        auc = roc_auc_score(y_valid_fold, Y_score)

        print('ROC/AUC Score: ', auc)
        print('Precision: ',precision )
        print('Recall: ', recall)

        print('\n')


        # print("%s: %.2f%%" % (cnn.GetModel().metrics_names[1], acc * 100))
        cvscores.append(auc)


    print("CV Score: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    total_seconds = time.time() - start_time
    print('CV Time: ', str(datetime.timedelta(seconds=total_seconds)) )

    return {'loss': -auc, 'status': STATUS_OK}


# def create_lstm(X_train, y_train, X_test, y_test):
#
#     lstm = LSTM_Model(output_dim=1, sequence_length=X_train.shape[1],
#                     nb_lstm_layers={{choice([1, 2, 3])}}, nb_units={{choice([5, 10, 15])}},
#                     activation={{choice(['relu', 'prelu'])}},
#                     dropout={{uniform(0, 1)}})
#
#     lstm.Build()
#
#     lstm.Compile(loss='binary_crossentropy',
#                 optimizer={{choice([SGD(lr=0.1, momentum=0.25, decay=0.0001, nesterov=True)])}},
#                 metrics=['accuracy'])
#
#     # reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001, verbose=1)
#
#     earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='auto')
#     checkpointer = ModelCheckpoint(filepath='lstm_tuned.hdf5',
#                                    verbose=0,
#                                    save_best_only=True)
#
#
#     # hist = model.FitData(X_train, y_train, batch_size=batch_size, nb_epochs=nb_epochs, cb1=reduceLR)
#     batch_size = {{choice([16, 32, 64, 128])}}
#
#     lstm.FitData(X_train=X_train, y_train=y_train, validation_split=0.08,
#                 batch_size=batch_size, nb_epochs=50, verbose=2, cb1=earlyStopping, cb2=checkpointer)
#     # cnn.FitDataWithValidation(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
#     #                                       batch_size=batch_size, nb_epochs=1, verbose=2)
#
#     score, acc = lstm.Evaluate(X_test, y_test, batch_size, verbose=0)
#
#     print('Test Accuracy: ', acc)
#
#     evaluator = ModelEvaluator(lstm, X_test=X_test, y_test=y_test, batch_size=32, generate_plots=False)
#
#
#     return {'loss': -acc, 'status': STATUS_OK, 'model': lstm.GetModel()}


def run_trials(model_type, evals=5):

    start_time = time.time()

    trials = Trials()

    if model_type == 'cnn':
        best_model = fmin(create_cnn_model, space, algo=tpe.suggest, max_evals=evals, trials=trials)
    elif model_type == 'lstm':
        pass

    total_seconds = time.time() - start_time
    print('Hyperparameter Optimization Time: ', str(datetime.timedelta(seconds=total_seconds)) )
    print('')

    # print("Best performing model chosen hyper-parameters:")
    # print(best_run)

    # json.dump(best_run, open("models/best_run.txt", 'w'))


    best_model_config = eval_hyperopt_space(space, best_model)

    print ('Best Configuration: ', best_model_config)


    return best_model_config