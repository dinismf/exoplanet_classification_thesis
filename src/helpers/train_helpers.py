from keras.api.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score
import time
import datetime
import numpy as np


def train_model(model, X_train, y_train, X_test, y_test, model_type='cnn', use_cv=False, nb_cv=5, batch_size=16,
                nb_epochs=40, save_name=''):
    """
    Generalized function to train a model either using cross-validation (CV) or standard training.

    Args:
        model: Keras model (CNNModel or LSTMModel)
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        model_type: Specify the type of model ('cnn' or 'lstm')
        use_cv: If True, use cross-validation. Otherwise, use standard training
        nb_cv: Number of cross-validation folds (if CV is used)
        batch_size: Batch size for training
        nb_epochs: Number of epochs for training
        save_name: Name to save the model (optional)

    Returns:
        None
    """

    if use_cv:
        # Cross-validation logic
        start_time = time.time()
        kfold = StratifiedKFold(n_splits=nb_cv, shuffle=True, random_state=7)
        cvscores = []

        for train_index, valid_index in kfold.split(X_train, y_train):
            X_train_fold = X_train[train_index]
            X_valid_fold = X_train[valid_index]
            y_train_fold = y_train[train_index]
            y_valid_fold = y_train[valid_index]

            # Reshape for CNN
            if model_type == 'cnn':
                X_train_fold = X_train_fold.reshape(X_train_fold.shape[0], X_train_fold.shape[1], 1)
                X_valid_fold = X_valid_fold.reshape(X_valid_fold.shape[0], X_valid_fold.shape[1], 1)

            # Train
            model.fit(X_train_fold, y_train_fold, batch_size=batch_size, epochs=nb_epochs, verbose=2)

            # Evaluate
            loss, acc = model.evaluate(X_valid_fold, y_valid_fold, batch_size=batch_size, verbose=0)

            # Predictions and metrics
            Y_score = model.predict(X_valid_fold)
            Y_predict = np.argmax(Y_score, axis=1)
            auc = roc_auc_score(y_valid_fold, Y_score)
            recall = recall_score(y_valid_fold, Y_predict)
            precision = precision_score(y_valid_fold, Y_predict)

            print(f"\nAccuracy: {acc}")
            print(f"ROC/AUC Score: {auc}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}\n")

            # Append CV score
            cvscores.append(auc)

        print(f"CV AUC Score: {np.mean(cvscores):.2f} (+/- {np.std(cvscores):.2f})")
        total_seconds = time.time() - start_time
        print(f"Total CV Time: {str(datetime.timedelta(seconds=total_seconds))}")

    else:
        # Standard training (without CV)
        print("Training without cross-validation...")
        start_time = time.time()

        # Train the model
        model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs,
                  verbose=2)

        # Evaluate the model
        loss, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)

        print(f"\nTest Accuracy: {acc}")
        print(f"Test Loss: {loss}")

        total_seconds = time.time() - start_time
        print(f"Total Training Time: {str(datetime.timedelta(seconds=total_seconds))}")

        return loss, acc

