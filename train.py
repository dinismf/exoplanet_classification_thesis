from data import LoadData
import preprocessing
from model import LSTM_Model, CNN_Model


def train_lstm(nn_type, batch_size, nb_layers, nb_hidden, dropout, nb_epoch, optimizer, activation, save_model=False, plot_loss=False, plot_data=[], filename=''):


    # Load original data
    data = LoadData()

    # Or load pre-processed version of original

    # Scaling/Normalization

    # Reshape input


    if (nn_type == 'lstm'):
        model = LSTM_Model(nb_layers=nb_layers, nb_units=nb_hidden, sequence_length=4767, output_dim=1, activation=activation, )



