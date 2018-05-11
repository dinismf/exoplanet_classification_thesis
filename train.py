from data import *
from model import LSTM_Model, CNN_Model
from evaluate import ModelEvaluator
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, precision_score
import datetime


def train_lstm(model, X_train, y_train, X_val, y_val, X_test, y_test, batch_size = 16, nb_epochs = 40, test_eval=True, save=False):


    history = model.FitDataWithValidation(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, batch_size=batch_size, nb_epochs=nb_epochs, verbose=1)

    if test_eval:
        print('Evaluating model on unseen test data...')
        evaluator = ModelEvaluator(model, X_test=X_test, y_test=y_test, batch_size=batch_size)
    else:
        print('Evaluating model on validation data...')
        evaluator = ModelEvaluator(model, X_test=X_val, y_test=y_val, batch_size=batch_size)

    evaluator.PlotTrainingPerformanceFromHistory(history)


def train_cnn(model, X_train, y_train, X_val, y_val, X_test, y_test, batch_size = 16, nb_epochs = 40, test_eval=True, save=False):


    reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001, verbose=1)
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=4, mode='auto')

    #history = model.FitData(X_train, y_train, batch_size=batch_size, nb_epochs=nb_epochs)
    history = model.FitDataWithValidation(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, batch_size=batch_size, nb_epochs=nb_epochs, verbose=1)
    #history = model.FitDataWithValidationCallbacks(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, batch_size=batch_size, nb_epochs=nb_epochs, cb1=reduceLR, cb2=earlyStopping, verbose=1)


    if test_eval:
        print('Evaluating model on unseen test data...')
        evaluator = ModelEvaluator(model, X_test=X_test, y_test=y_test, batch_size=batch_size)
    else:
        print('Evaluating model on validation data...')
        evaluator = ModelEvaluator(model, X_test=X_val, y_test=y_val, batch_size=batch_size)

    evaluator.PlotTrainingPerformanceFromHistory(history)

def train_cnn_cv(model, X_train, y_train, X_test, y_test, nb_cv = 5, batch_size = 16, nb_epochs = 40, save_name = ''):

    start_time = time.time()

    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=nb_cv, shuffle=True, random_state=7)

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

        # reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001, verbose=1)
        # earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto')
        model.FitData(X_train=X_train_fold, y_train=y_train_fold, batch_size=batch_size, nb_epochs=nb_epochs, verbose=2)
        # cnn.FitDataWithValidationCallbacks(X_train=X[train], y_train=y[train], X_val=X[valid], y_val=y[valid],
        #                                   batch_size=batch_size, nb_epochs=50, verbose=2, cb1=earlyStopping)

        score, acc = model.Evaluate(X_valid_fold, y_valid_fold, batch_size, verbose=0)

        Y_score, Y_predict, Y_true = model.Predict(X_valid_fold, y_valid_fold)

        auc = roc_auc_score(y_valid_fold, Y_score)
        recall = recall_score(y_valid_fold, Y_predict)
        precision = precision_score(y_valid_fold, Y_predict)

        print('\n')
        print('Acc: ', acc)
        print('ROC/AUC Score: ', auc)
        print('Precision: ',precision )
        print('Recall: ', recall)
        print('\n')

        # print("%s: %.2f%%" % (cnn.GetModel().metrics_names[1], acc * 100))
        cvscores.append(auc)

    print("CV Score: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    total_seconds = time.time() - start_time
    print('CV Time: ', str(datetime.timedelta(seconds=total_seconds)))


    if (save_name != ''):
        model.SaveCNNModel(save_name, config=True, weights=True)


    evaluator = ModelEvaluator(model, X_test=X_test, y_test=y_test, batch_size=batch_size, generate_plots=True)

    evaluator.PlotTrainingPerformance(history)


def train_lstm_cv(model, X_train, y_train, X_test, y_test, nb_cv = 5, batch_size = 16, nb_epochs = 40, save_name = ''):

    start_time = time.time()

    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=nb_cv, shuffle=True, random_state=7)

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

        # reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001, verbose=1)
        # earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto')
        model.FitData(X_train=X_train_fold, y_train=y_train_fold, batch_size=batch_size, nb_epochs=nb_epochs, verbose=2)
        # cnn.FitDataWithValidationCallbacks(X_train=X[train], y_train=y[train], X_val=X[valid], y_val=y[valid],
        #                                   batch_size=batch_size, nb_epochs=50, verbose=2, cb1=earlyStopping)

        score, acc = model.Evaluate(X_valid_fold, y_valid_fold, batch_size, verbose=0)

        Y_score, Y_predict, Y_true = model.Predict(X_valid_fold, y_valid_fold)

        auc = roc_auc_score(y_valid_fold, Y_score)
        recall = recall_score(y_valid_fold, Y_predict)
        precision = precision_score(y_valid_fold, Y_predict)

        print('\n')
        print('Acc: ', acc)
        print('ROC/AUC Score: ', auc)
        print('Precision: ',precision )
        print('Recall: ', recall)
        print('\n')

        # print("%s: %.2f%%" % (cnn.GetModel().metrics_names[1], acc * 100))
        cvscores.append(auc)

    print("CV Score: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    total_seconds = time.time() - start_time
    print('CV Time: ', str(datetime.timedelta(seconds=total_seconds)))


    if (save_name != ''):
        model.SaveLSTMModel(save_name, config=True, weights=True)


    evaluator = ModelEvaluator(model, X_test=X_test, y_test=y_test, batch_size=batch_size, generate_plots=True)

    evaluator.PlotTrainingPerformance(history)


# if __name__ == "__main__":




