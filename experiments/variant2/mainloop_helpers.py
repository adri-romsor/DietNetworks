from __future__ import print_function
import numpy as np
import os
import random

from feature_selection.experiments.common import dataset_utils as du
from feature_selection.experiments.common import imdb, dragonn_data


# Function to load data
def load_data(dataset, dataset_path, embedding_source,
              which_fold=0, keep_labels=1., missing_labels_val=1.):

    # Load data from specified dataset
    splits = [.6, .2]  # this will split the data into [60%, 20%, 20%]
    if dataset == 'protein_binding':
        data = du.load_protein_binding(transpose=False, splits=splits)
    elif dataset == 'dorothea':
        data = du.load_dorothea(transpose=False, splits=None)
    elif dataset == 'opensnp':
        data = du.load_opensnp(transpose=False, splits=splits)
    elif dataset == 'reuters':
        data = du.load_reuters(transpose=False, splits=splits)
    elif dataset == 'iric_molecule':
        data = du.load_iric_molecules(
            transpose=False, splits=splits)
    elif dataset == 'imdb':
        dataset_path = os.path.join(dataset_path, "imdb")
        feat_type = "tfidf"
        unsupervised = False

        print ("Loading imdb")
        if unsupervised:
            file_name = os.path.join(
                dataset_path,
                'unsupervised_IMDB_' + feat_type + '_table_split80.hdf5')
        else:
            file_name = os.path.join(
                dataset_path,
                'supervised_IMDB_' + feat_type + '_table_split80.hdf5')

        # This is in order to copy dataset if it doesn't exist
        print (file_name)
        print (os.path.isfile(file_name))

        if not os.path.isfile(file_name):
            print ("Saving dataset to path")
            imdb.save_as_hdf5(
                path=dataset_path,
                unsupervised=unsupervised,
                feat_type=feat_type,
                use_tables=False)
            print ("Done saving dataset")
        # use feat_type='tfidf' to load tfidf features
        data = imdb.read_from_hdf5(
            path=dataset_path, unsupervised=unsupervised, feat_type=feat_type)
    elif dataset == 'dragonn':
        data = dragonn_data.load_data(500, 100, 100)
    elif dataset == '1000_genomes':
        # This will split the training data into 75% train, 25%
        # this corresponds to the split 60/20 of the whole data,
        # test is considered elsewhere as an extra 20% of the whole data
        splits = [.75]
        data = du.load_1000_genomes(transpose=False,
                                    label_splits=splits,
                                    fold=which_fold)
    else:
        print("Unknown dataset")
        return

    if dataset == 'imdb':
        x_train = data.root.train_features
        y_train = data.root.train_labels[:][:, None].astype("float32")
        x_valid = data.root.val_features
        y_valid = data.root.val_labels[:][:, None].astype("float32")
        x_test = data.root.test_features
        y_test = None
        x_nolabel = None
    else:
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test),\
            x_nolabel = data

    if not embedding_source:
        if x_nolabel is None:
            if dataset == 'imdb':
                x_unsup = x_train[:5000].transpose()
            else:
                x_unsup = x_train.transpose()
        else:
            x_unsup = np.vstack((x_train, x_nolabel)).transpose()
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
                    n_hidden_t_enc, n_hidden_t_dec, n_hidden_s, which_fold):
    # Define experiment name from parameters
    exp_name = 'our_model' + str(keep_labels) + \
        ('_Ri' if gamma > 0 else '') + ('_Rwenc' if alpha > 0 else '') + \
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
               prec_recall_cutoff=True):
    print('-'*20 + which_set + ' monit.' + '-'*20)
    monitoring_values = np.zeros(len(monitoring_labels), dtype="float32")
    global_batches = 0

    targets = []
    predictions = []

    for batch in minibatches:
        # Update monitored values
        out = error_fn(*batch)

        monitoring_values = monitoring_values + out[1:]
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

    return monitoring_values
