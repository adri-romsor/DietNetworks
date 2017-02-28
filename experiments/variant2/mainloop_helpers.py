from __future__ import print_function
import numpy as np
import os
import random
from DietNetworks.experiments.common import dataset_utils as du

# Function to load data
def load_data(dataset, dataset_path, embedding_source,
              which_fold=0, keep_labels=1., missing_labels_val=1.,
              embedding_input='raw', transpose=False, norm=True):

    # Load data from specified dataset
    splits = [.6, .2]  # this will split the data into [60%, 20%, 20%]
    if dataset == '1000_genomes':
        # This will split the training data into 75% train, 25%
        # this corresponds to the split 60/20 of the whole data,
        # test is considered elsewhere as an extra 20% of the whole data
        splits = [.75]
        data = du.load_1000_genomes(transpose=transpose,
                                    label_splits=splits,
                                    feature_splits=[.8],
                                    fold=which_fold,
                                    nolabels=embedding_input,
                                    norm=norm, path=dataset_path)
    else:
        print("Unknown dataset")
        return

    if not transpose:
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test),\
            x_nolabel = data
    else:
        return data

    if not embedding_source:
        if x_nolabel is None:
            x_unsup = x_train.transpose()
        else:
            x_unsup = x_nolabel
    else:
        x_unsup = None

    # If needed, remove some of the training labels
    if keep_labels <= 1.0:
        training_labels = y_train.copy()
        random.seed(23)
        nb_train = len(training_labels)

        indices = range(nb_train)
        random.shuffle(indices)

        indices_discard = indices[:int(nb_train * (1 - keep_labels))]
        for idx in indices_discard:
            training_labels[idx] = missing_labels_val
    else:
        training_labels = y_train

    return x_train, y_train, x_valid, y_valid, x_test, y_test, \
        x_unsup, training_labels


def define_exp_name(keep_labels, alpha, beta, gamma, lmd, n_hidden_u,
                    n_hidden_t_enc, n_hidden_t_dec, n_hidden_s, which_fold,
                    embedding_input, lr, dni, eni, earlystop, anneal):
    # Define experiment name from parameters
    exp_name = 'our_model' + str(keep_labels) + \
        '_' + embedding_input + '_lr-' + str(lr) + '_anneal-' + str(anneal)  +\
        ('_eni-' + str(eni) if eni > 0 else '') + \
        ('_dni-' + str(dni) if dni > 0 else '') + \
        '_' + earlystop + \
        ('_Ri'+str(gamma) if gamma > 0 else '') + ('_Rwenc' if alpha > 0 else '') + \
        ('_Rwdec' if beta > 0 else '') + \
        (('_l2-' + str(lmd)) if lmd > 0. else '')
    exp_name += '_hu'
    for i in range(len(n_hidden_u)):
        exp_name += ("-" + str(n_hidden_u[i]))
    exp_name += '_tenc'
    for i in range(len(n_hidden_t_enc)):
        exp_name += ("-" + str(n_hidden_t_enc[i]))
    exp_name += '_tdec'
    for i in range(len(n_hidden_t_dec)):
        exp_name += ("-" + str(n_hidden_t_dec[i]))
    exp_name += '_hs'
    for i in range(len(n_hidden_s)):
        exp_name += ("-" + str(n_hidden_s[i]))
    exp_name += '_fold' + str(which_fold)

    return exp_name


# Mini-batch iterator function
def iterate_minibatches(inputs, targets, batchsize,
                        shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    indices = np.arange(inputs.shape[0])
    if shuffle:
        indices = np.random.permutation(inputs.shape[0])

    for i in range(0, inputs.shape[0]-batchsize+1, batchsize):
        yield inputs[indices[i:i+batchsize], :],\
            targets[indices[i:i+batchsize]]

def iterate_minibatches_unsup(x, batch_size, shuffle=False):
    indices = np.arange(x.shape[0])
    if shuffle:
        indices = np.random.permutation(x.shape[0])
    for i in range(0, x.shape[0]-batch_size+1, batch_size):
        yield x[indices[i:i+batch_size], :]


def iterate_testbatches(inputs, batchsize, shuffle=False):
    indices = np.arange(inputs.shape[0])
    if shuffle:
        indices = np.random.permutation(inputs.shape[0])
    for i in range(0, inputs.shape[0]-batchsize+1, batchsize):
        yield inputs[indices[i:i+batchsize], :]


def get_precision_recall_cutoff(predictions, targets):

    prev_threshold = 0.00
    threshold_inc = 0.10

    while True:
        if prev_threshold > 1.000:
            cutoff = 0.0
            break

        threshold = prev_threshold + threshold_inc
        tp = ((predictions >= threshold) * (targets == 1)).sum()
        fp = ((predictions >= threshold) * (targets == 0)).sum()
        fn = ((predictions < threshold) * (targets == 1)).sum()

        precision = float(tp) / (tp + fp + 1e-20)
        recall = float(tp) / (tp + fn + 1e-20)

        if precision > recall:
            if threshold_inc < 0.001:
                cutoff = recall
                break
            else:
                threshold_inc /= 10
        else:
            prev_threshold += threshold_inc

    return cutoff


# Monitoring function
def monitoring(minibatches, which_set, error_fn, monitoring_labels,
               prec_recall_cutoff=True, start=1, return_pred=False):
    print('-'*20 + which_set + ' monit.' + '-'*20)
    prec_recall_cutoff = False if start == 0 else prec_recall_cutoff
    monitoring_values = np.zeros(len(monitoring_labels), dtype="float32")
    global_batches = 0

    targets = []
    predictions = []

    for batch in minibatches:
        # Update monitored values
        if start == 0:
            out = error_fn(batch)
        else:
            out = error_fn(*batch)

        monitoring_values = monitoring_values + out[start:]
        if start == 1:
            predictions.append(out[0])
        targets.append(batch[1])
        global_batches += 1

    # Print monitored values
    monitoring_values /= global_batches
    for (label, val) in zip(monitoring_labels, monitoring_values):
        print ("  {} {}:\t\t{:.6f}".format(which_set, label, val))

    # If needed, compute and print the precision-recall breakoff point
    if prec_recall_cutoff:
        predictions = np.vstack(predictions)
        targets = np.vstack(targets)
        cutoff = get_precision_recall_cutoff(predictions, targets)
        print ("  {} precis/recall cutoff:\t{:.6f}".format(which_set, cutoff))

    if return_pred:
        return monitoring_values, np.vstack(predictions), np.vstack(targets)
    else:
        return monitoring_values


def parse_int_list_arg(arg):
    if isinstance(arg, str):
        arg = eval(arg)

    if isinstance(arg, list):
        return arg
    if isinstance(arg, int):
        return [arg]
    else:
        raise ValueError("Following arg value could not be cast as a list of"
                         "integer values : " % arg)


def parse_string_int_tuple(arg):
    if isinstance(arg, (list, tuple)):
        return arg
    elif isinstance(arg, str):
        tmp = arg.strip("()[]").split(",")
        assert (len(tmp) == 2)
        return (tmp[0], eval(tmp[1]))
    else:
        raise NotImplementedError()
