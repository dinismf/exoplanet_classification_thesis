import configparser
import time
import json
import train
import os.path

if __name__ == '__main__':

    config_filename = 'experiment.ini'

    if not os.path.isfile(config_filename):
        raise Exception('File does not exist')

    # Read the experiment configuration
    config = configparser.ConfigParser()
    config.read(config_filename)

    # Retrieve network parameters
    nn_type = config.get('Network', 'nn_type')
    nb_hidden = config.get('Network', 'hidden_layers')
    dropout = config.get('Network', 'dropout')

    # Retrieve training parameters

    # Retrieve data parameters

    # Retrieve other options
    root_filename = config.get('Options', 'filename')
    plot_loss = config.getboolean('Options', 'plot_loss')
    save_model = config.getboolean('Options', 'save_model')

    start_time = time.time()

    # Train network

    tr
    ain

    print('Training time (seconds): ', (time.time() - start_time))




