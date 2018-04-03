import configparser
import time
import json
import os
import os.path
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras import backend as K
K.tensorflow_backend._get_available_gpus()
from train import *

if __name__ == '__main__':

    #config_filename = 'experiment_lstm.ini'
    config_filename = 'experiment_cnn.ini'

    if not os.path.isfile(config_filename):
        raise Exception('File does not exist')

    # Read the experiment configuration
    config = configparser.ConfigParser()
    config.read(config_filename)

    # Retrieve network parameters
    nn_type = config.get('Network', 'nn_type')
   # nb_layers = config.get('Network', 'nb_layers')
    nb_hidden = config.get('Network', 'hidden_units')
    activation = config.get('Network', 'activation')
    dropout = config.get('Network', 'dropout')

    # Retrieve training parameters

    # Retrieve data parameters
    data_path = config.get('Data', 'data')
    preprocessing_na = config.get('Data', 'preprocessing_na')
    preprocessing_scale = config.get('Data', 'preprocessing_scale')

    # Retrieve other options
    root_filename = config.get('Options', 'filename')
    plot_loss = config.getboolean('Options', 'plot_loss')
    save_model = config.getboolean('Options', 'save_model')

    start_time = time.time()

    # Train network
    if (nn_type == 'LSTM'):
        train_lstm()

    elif (nn_type == 'CNN'):
        train_cnn()

    print('Training time (seconds): ', (time.time() - start_time))




