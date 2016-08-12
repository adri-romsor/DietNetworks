import sys

import random
from collections import namedtuple

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import dragonn.simulations

Data = namedtuple('Data', ['X_train', 'X_valid', 'X_test',
                           'y_train', 'y_valid', 'y_test',
                           'motif_names'])
random.seed(1)


def get_available_simulations():
    return [function_name for function_name in dir(dragonn.simulations)
            if "simulate" in function_name]


def print_available_simulations():
    for function_name in get_available_simulations():
        print function_name


def get_simulation_function(simulation_name):
    if simulation_name in get_available_simulations():
        return getattr(dragonn.simulations, simulation_name)
    else:
        print("%s is not available. Available simulations are:" %
              (simulation_name))
        print_available_simulations()


def print_simulation_info(simulation_name):
    simulation_function = get_simulation_function(simulation_name)
    if simulation_function is not None:
        print simulation_function.func_doc


def get_simulation_data(simulation_name, simulation_parameters,
                        test_set_size=4000, validation_set_size=3200):
    simulation_function = get_simulation_function(simulation_name)
    try:
        sequences, y = simulation_function(**simulation_parameters)
    except Exception as e:
        return e

    if simulation_name == "simulate_heterodimer_grammar":
        motif_names = [simulation_parameters["motif1"],
                       simulation_parameters["motif2"]]
    elif simulation_name == "simulate_multi_motif_embedding":
        motif_names = simulation_parameters["motif_names"]
    else:
        motif_names = [simulation_parameters["motif_name"]]

    train_sequences, test_sequences, y_train, y_test = train_test_split(
        sequences, y, test_size=test_set_size)
    train_sequences, valid_sequences, y_train, y_valid = train_test_split(
        train_sequences, y_train, test_size=validation_set_size)
    X_train = one_hot_encode(train_sequences)
    X_valid = one_hot_encode(valid_sequences)
    X_test = one_hot_encode(test_sequences)

    X_train = get_matrix(X_train)
    X_valid = get_matrix(X_valid)
    X_test = get_matrix(X_test)

    return Data(X_train, X_valid, X_test,
                y_train, y_valid, y_test, motif_names)


def one_hot_encode(sequences):
    sequence_length = len(sequences[0])
    integer_type = np.int8 if sys.version_info[
        0] == 2 else np.int32  # depends on Python version
    integer_array = \
        LabelEncoder().fit(np.array(('ACGTN',)).view(integer_type)).transform(
            sequences.view(integer_type)).reshape(len(sequences),
                                                  sequence_length)
    one_hot_encoding = OneHotEncoder(
        sparse=False, n_values=5).fit_transform(integer_array)

    return one_hot_encoding.reshape(
        len(sequences), 1,
        sequence_length, 5).swapaxes(2, 3)[:, :, [0, 1, 2, 4], :]


def get_sequence_strings(encoded_sequences):
    """
    Converts encoded sequences into an array with sequence strings
    """
    num_samples, _, _, seq_length = np.shape(encoded_sequences)
    sequence_characters = np.chararray((num_samples, seq_length))
    sequence_characters[:] = 'N'
    for i, letter in enumerate(['A', 'C', 'G', 'T']):
        letter_indxs = (encoded_sequences[:, :, i, :] == 1).squeeze()
        sequence_characters[letter_indxs] = letter
    # return 1D view of sequence characters
    return sequence_characters.view('S%s' % (seq_length)).ravel()


def get_matrix(X):
    X = np.transpose(X, (0, 1, 3, 2))
    sh = X.shape
    X = np.reshape(X, (sh[0], np.prod(sh[1:])))

    return X


def load_data(seq_len, num_pos, num_neg):
    heterodimer_grammar_simulation_parameters = {"seq_length": seq_len,
                                                 "GC_fraction": 0.4,
                                                 "num_pos": num_pos,
                                                 "num_neg": num_neg,
                                                 "motif1": "SPI1_known4",
                                                 "motif2": "IRF_known20",
                                                 "min_spacing": 2,
                                                 "max_spacing": 5}
    simulation_data = \
        get_simulation_data("simulate_heterodimer_grammar",
                            heterodimer_grammar_simulation_parameters)

    return simulation_data

if __name__ == "__main__":
    sd = load_data(500, 10000, 10000)
