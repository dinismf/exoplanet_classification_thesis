import os
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from src.helpers.train_helpers import train_model
from src.helpers.import_helpers import load_dataset
from src.models.model import CNNModel, LSTMModel
from keras.api.optimizers import Adam
from keras.api.models import load_model
from definitions import MODELS_OUTPUT_DIR
from pathlib import Path

import json

# CNN Hyperparameter Space
cnn_space = {
    'nb_blocks': hp.choice('nb_blocks', [0, 1, 2, 3]),
    'filters': hp.choice('filters', [8, 16, 32, 64, 128, 254]),
    'kernel_size': hp.choice('kernel_size', [3, 5, 7, 9]),
    'pooling': hp.choice('pooling', ['max', 'average']),
    'pooling_size': hp.choice('pooling_size', [2, 3, 4]),
    'pooling_strides': hp.choice('pooling_strides', [2, 3, 4]),
    'conv_dropout': hp.uniform('conv_dropout', 0.0, 0.35),
    'fc_dropout': hp.uniform('fc_dropout', 0.0, 0.6),
    'fc_units': hp.choice('fc_units', [32, 64, 128, 254]),
    'batch_size': hp.choice('batch_size', [16, 32]),
    'lr_rate_mult': hp.loguniform('lr_rate_mult', -0.5, 0.5),
    'momentum': hp.choice('momentum', [0, 0.25, 0.4]),
    'batch_norm': hp.choice('batch_norm', [True, False]),
    'nb_epochs': hp.uniform('nb_epochs', 5.0, 50.0),
    'activation': 'relu'
}

# LSTM Hyperparameter Space
lstm_space = {
    'nb_lstm_layers': hp.choice('nb_lstm_layers', [0, 1]),
    'lstm_units': hp.choice('lstm_units', [2, 5, 10, 15]),
    'dropout': hp.uniform('dropout', 0.0, 0.5),
    'fc_units': hp.choice('fc_units', [32, 64, 128]),
    'fc_layers': hp.choice('fc_layers', [0, 1, 2]),
    'batch_size': hp.choice('batch_size', [16, 32, 64]),
    'lr_rate_mult': hp.loguniform('lr_rate_mult', -0.5, 0.5),
    'momentum': hp.choice('momentum', [0, 0.25, 0.4]),
    'batch_norm': hp.choice('batch_norm', [True, False]),
    'nb_epochs': hp.uniform('nb_epochs', 10.0, 50.0),
    'activation': hp.choice('activation', ['relu'])
}


def cnn_objective(params, X_train, y_train, X_test, y_test):
    # Build CNN model based on 'params'
    model = CNNModel(
        filters=params['filters'],
        kernel_size=params['kernel_size'],
        pooling=params['pooling'],
        pooling_size=params['pooling_size'],
        pooling_strides=params['pooling_strides'],
        conv_dropout=params['conv_dropout'],
        fc_dropout=params['fc_dropout'],
        fc_units=params['fc_units'],
        nb_blocks=params['nb_blocks'],
        batch_norm=params['batch_norm'],
        activation=params['activation']
    )

    # Compile and train model
    model.model.compile(optimizer=Adam(learning_rate=params['lr_rate_mult']),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=params['batch_size'], epochs=int(params['nb_epochs']), verbose=0)

    # Evaluate the model
    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    # Return the loss for Hyperopt to minimize, and store the model as part of the return value
    return {'loss': loss, 'status': STATUS_OK, 'model': model}


def lstm_objective(params, X_train, y_train, X_test, y_test):
    # Create LSTM model using params
    model = LSTMModel(
        nb_lstm_layers=params['nb_lstm_layers'],
        lstm_units=params['lstm_units'],
        dropout=params['dropout'],
        fc_units=params['fc_units'],
        fc_layers=params['fc_layers'],
        batch_norm=params['batch_norm'],
        activation=params['activation']
    )

    # Compile the model directly in train_model.py
    optimizer = Adam(learning_rate=params['lr_rate_mult'])
    model.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Train and evaluate the model
    loss, accuracy = train_model(model, X_train, y_train, X_test, y_test, model_type='lstm', use_cv=False,
                                 nb_epochs=int(params['nb_epochs']), batch_size=params['batch_size'])
    return {'loss': loss, 'status': STATUS_OK}


def save_model_and_config(model, model_type, dataset_name, params=None):
    """
    Save the trained model and its configuration (hyperparameters) for future reference.

    Args:
        model: Trained Keras model
        model_type: Either 'cnn' or 'lstm'
        dataset_name: The dataset used for training
        params: Hyperparameters or model configuration
    """

    # Create a directory specific to the model type
    save_dir = MODELS_OUTPUT_DIR / model_type
    save_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

    # Remove the extension from the dataset name
    dataset_name = Path(dataset_name).stem

    # Use a descriptive subdirectory or file name for the model
    model_save_path = save_dir / f"{model.name}_{dataset_name}.keras"

    # Save model
    model.save(str(model_save_path))

    # Save the configuration (hyperparameters or model info)
    if params:
        config_name = f"{model.name}_{dataset_name}_config.json"
        config_path = save_dir / config_name
        with open(config_path, 'w') as f:
            json.dump(params, f, indent=4)

    print(f"Model and configuration saved to {model_save_path}")


def run_cnn_optimization(X_train, y_train, X_test, y_test):
    trials = Trials()
    best_params = fmin(fn=lambda params: cnn_objective(params, X_train, y_train, X_test, y_test), space=cnn_space,
                       algo=tpe.suggest,
                       max_evals=50, trials=trials)
    print("Best CNN hyperparameters found: ", best_params)
    return best_params


def run_lstm_optimization(X_train, y_train, X_test, y_test):
    trials = Trials()
    best_params = fmin(fn=lambda params: lstm_objective(params, X_train, y_train, X_test, y_test), space=lstm_space,
                       algo=tpe.suggest,
                       max_evals=50, trials=trials)
    print("Best LSTM hyperparameters found: ", best_params)
    return best_params


def load_existing_model(model_path):
    """
    Load an existing model from the filesystem.

    Args:
        model_path: Path to the saved model.

    Returns:
        Loaded Keras model.
    """
    if os.path.isfile(model_path):
        print(f"Loading model from {model_path}...")
        return load_model(model_path)
    else:
        raise FileNotFoundError(f"Model at {model_path} not found.")


def retrain_existing_model(model, X_train, y_train, X_test, y_test, batch_size=16, nb_epochs=40):
    """
    Retrain the existing model with new or additional data.

    Args:
        model: Loaded Keras model to be retrained.
        X_train: Training features.
        y_train: Training labels.
        X_test: Testing features.
        y_test: Testing labels.
        batch_size: Batch size for training.
        nb_epochs: Number of epochs for retraining.

    Returns:
        Retrained model.
    """
    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, validation_data=(X_test, y_test))
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nRetrained Model - Test Loss: {loss}, Test Accuracy: {accuracy}")
    return model


def train_model_pipeline(model_type='cnn', use_cv=False, optimize=False, nb_cv=5, batch_size=16, nb_epochs=40,
                         dataset_name='', save_name='', retrain=False, existing_model_path=None):
    """
    Pipeline to train or retrain a model, with options for cross-validation, standard training, and hyperparameter optimization.

    Args:
        model_type: Either 'cnn' or 'lstm'
        use_cv: Whether to use cross-validation
        optimize: Whether to run hyperparameter optimization
        nb_cv: Number of CV folds (if using cross-validation)
        batch_size: Batch size
        nb_epochs: Number of epochs
        dataset_name: Pre-processed dataset to use for training
        save_name: Name to save the model
        retrain: Boolean flag to retrain an existing model.
        existing_model_path: Path to the existing model for retraining.
    """

    # Load data
    X_train, X_test, y_train, y_test = load_dataset(dataset_name=dataset_name)

    # Reshape data for CNN if necessary
    if model_type == 'cnn' or model_type == 'lstm':
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    if retrain:
        # Load and retrain the existing model
        if existing_model_path:
            model = load_existing_model(existing_model_path)
            model = retrain_existing_model(model, X_train, y_train, X_test, y_test, batch_size=batch_size,
                                           nb_epochs=nb_epochs)
            # Save the retrained model
            save_model_and_config(model, model_type, dataset_name)
        else:
            print("Error: No path provided for the existing model.")
            return

    elif optimize:
        # Run optimization based on model type
        if model_type == 'cnn':
            best_params = run_cnn_optimization(X_train, y_train, X_test, y_test)
        elif model_type == 'lstm':
            best_params = run_lstm_optimization(X_train, y_train, X_test, y_test)

        # Save the model and configuration
        save_model_and_config(model=None, model_type=model_type, dataset_name=dataset_name, params=best_params)

    else:
        # Standard or cross-validation training
        if model_type == 'cnn':
            model = CNNModel(
                filters=32,
                kernel_size=3,
                pooling='max',
                pooling_size=2,
                pooling_strides=2,
                conv_dropout=0.2,
                fc_dropout=0.5,
                fc_units=128,
                nb_blocks=2,
                batch_norm=True,
                activation='relu'
            )
        elif model_type == 'lstm':
            model = LSTMModel(
                nb_lstm_layers=1,
                lstm_units=64,
                dropout=0.3,
                fc_units=64,
                fc_layers=1,
                batch_norm=True,
                activation='relu'
            )

        # Compile the model directly in train_model.py
        optimizer = Adam(learning_rate=0.001)  # Use default learning rate, adjust as needed
        model.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Train with or without cross-validation
        train_model(model=model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model_type=model_type,
                    use_cv=use_cv, nb_cv=nb_cv,
                    batch_size=batch_size, nb_epochs=nb_epochs, save_name=save_name)

        # Save the final trained model (without optimization)
        save_model_and_config(model=model, model_type=model_type, dataset_name=dataset_name)


if __name__ == "__main__":
    model_type = input("Enter 'cnn' for CNN or 'lstm' for LSTM: ").strip().lower()
    retrain = input("Do you want to retrain an existing model? (y/n): ").strip().lower() == 'y'

    if retrain:
        existing_model_path = input("Enter the path of the model to retrain: ").strip()
        dataset_name = input("Enter the name of the dataset for retraining: ").strip()
        train_model_pipeline(model_type=model_type, retrain=True, existing_model_path=existing_model_path,
                             dataset_name=dataset_name)
    else:
        use_cv = input("Use cross-validation? (y/n): ").strip().lower() == 'y'
        optimize = input("Run hyperparameter optimization? (y/n): ").strip().lower() == 'y'
        dataset_name = input("Enter the name of the dataset to use: ").strip()
        train_model_pipeline(model_type=model_type, use_cv=use_cv, optimize=optimize, dataset_name=dataset_name)
