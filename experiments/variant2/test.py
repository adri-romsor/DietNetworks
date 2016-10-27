#!/usr/bin/env python2

from __future__ import print_function
import argparse
import time
import os
import random
from distutils.dir_util import copy_tree

import lasagne
from lasagne.layers import DenseLayer, InputLayer, DropoutLayer, BatchNormLayer
from lasagne.nonlinearities import (sigmoid, softmax, tanh, linear, rectify,
                                    leaky_rectify, very_leaky_rectify)
from lasagne.init import Uniform
import numpy as np
import theano
import theano.tensor as T

from feature_selection.experiments.common import dataset_utils, imdb

import matplotlib.pyplot as plt

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

    return monitoring_values, predictions, targets


# Main program
def execute(dataset, n_hidden_u, n_hidden_t_enc, n_hidden_t_dec, n_hidden_s,
            embedding_source=None,
            num_epochs=500, learning_rate=.001, learning_rate_annealing=1.0,
            gamma=1, disc_nonlinearity="sigmoid", encoder_net_init=0.2,
            decoder_net_init=0.2, keep_labels=1.0, prec_recall_cutoff=True,
            missing_labels_val=-1.0, early_stop_criterion='loss_sup_det',
            save_copy='/Tmp/romerosa/feature_selection/',
            dataset_path='/Tmp/' + os.environ["USER"] + '/datasets/'):

    # Load the dataset
    print("Loading data")
    splits = [0.6, 0.2]  # This will split the data into [60%, 20%, 20%]

    if dataset == 'protein_binding':
        data = dataset_utils.load_protein_binding(transpose=False,
                                                  splits=splits)
    elif dataset == 'dorothea':
        data = dataset_utils.load_dorothea(transpose=False, splits=None)
    elif dataset == 'opensnp':
        data = dataset_utils.load_opensnp(transpose=False, splits=splits)
    elif dataset == 'reuters':
        data = dataset_utils.load_reuters(transpose=False, splits=splits)
    elif dataset == 'iric_molecule':
        data = dataset_utils.load_iric_molecules(transpose=False, splits=splits)
    elif dataset == 'imdb':
        dataset_path = os.path.join(dataset_path,"imdb")
        # use feat_type='tfidf' to load tfidf features
        data = imdb.read_from_hdf5(path=dataset_path,unsupervised=False, feat_type='tfidf')
    elif dataset == 'dragonn':
        from feature_selection.experiments.common import dragonn_data
        data = dragonn_data.load_data(500, 100, 100)
    elif dataset == '1000_genomes':
        data = dataset_utils.load_1000_genomes(transpose=False,
                                               label_splits=splits)
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
        n_samples_unsup = x_unsup.shape[1]
    else:
        x_unsup = None

    # Extract required information from data
    n_samples, n_feats = x_train.shape
    n_targets = y_train.shape[1]

    # Set some variables
    batch_size = 128

    # Preparing folder to save stuff
    exp_name = 'our_model' + str(keep_labels) + '_sup' + \
        ('_unsup' if gamma > 0 else '')
    save_copy = os.path.join(save_copy, dataset, exp_name)

    # Prepare Theano variables for inputs and targets
    input_var_sup = T.matrix('input_sup')
    input_var_unsup = theano.shared(x_unsup, 'input_unsup')  # x_unsup TBD
    target_var_sup = T.matrix('target_sup')
    lr = theano.shared(np.float32(learning_rate), 'learning_rate')

    # Build model
    print("Building model")

    # Some checkings
    assert len(n_hidden_u) > 0
    assert len(n_hidden_t_enc) > 0
    assert len(n_hidden_t_dec) > 0
    assert len(n_hidden_u) > 0
    assert n_hidden_t_dec[-1] == n_hidden_t_enc[-1]

    # Build unsupervised network
    encoder_net = InputLayer((n_feats, n_samples_unsup), input_var_unsup)
    for i, out in enumerate(n_hidden_u):
        encoder_net = DenseLayer(encoder_net, num_units=out,
                                 nonlinearity=rectify)
    feat_emb = lasagne.layers.get_output(encoder_net)
    pred_feat_emb = theano.function([], feat_emb)

    # Build transformations (f_theta, f_theta') network and supervised network
    # f_theta (ou W_enc)
    encoder_net_W_enc = encoder_net
    for hid in n_hidden_t_enc:
        encoder_net_W_enc = DenseLayer(encoder_net_W_enc, num_units=hid,
                                       nonlinearity=tanh,
                                       W=Uniform(encoder_net_init)
                                       )
    enc_feat_emb = lasagne.layers.get_output(encoder_net_W_enc)

    # f_theta' (ou W_dec)
    encoder_net_W_dec = encoder_net
    for hid in n_hidden_t_dec:
        encoder_net_W_dec = DenseLayer(encoder_net_W_dec, num_units=hid,
                                       nonlinearity=tanh,
                                       W=Uniform(decoder_net_init)
                                       )
    dec_feat_emb = lasagne.layers.get_output(encoder_net_W_dec)

    # Supervised network
    discrim_net = InputLayer((batch_size, n_feats), input_var_sup)
    discrim_net = DenseLayer(discrim_net, num_units=n_hidden_t_enc[-1],
                             W=enc_feat_emb, nonlinearity=rectify)

    # reconstruct the input using dec_feat_emb
    reconst_net = DenseLayer(discrim_net, num_units=n_feats,
                             W=dec_feat_emb.T)

    # predicting labels
    for hid in n_hidden_s:
        discrim_net = DropoutLayer(discrim_net)
        discrim_net = DenseLayer(discrim_net, num_units=hid)

    assert disc_nonlinearity in ["sigmoid", "linear", "rectify", "softmax"]
    discrim_net = DropoutLayer(discrim_net)
    discrim_net = DenseLayer(discrim_net, num_units=n_targets,
                             nonlinearity=eval(disc_nonlinearity))

    print("Building and compiling training functions")
    # Build and compile training functions
    prediction_det = lasagne.layers.get_output(discrim_net,
                                               deterministic=True)

    inputs = [input_var_sup, target_var_sup]

    params = lasagne.layers.get_all_params([discrim_net, reconst_net,
                                            encoder_net_W_dec,
                                            encoder_net_W_enc],
                                           trainable=True)

    # Expressions required for test
    monitor_labels = []
    val_outputs = []

    if disc_nonlinearity in ["sigmoid", "softmax"]:
        if disc_nonlinearity == "sigmoid":
            test_pred = T.gt(prediction_det, 0.5)
            test_acc = T.mean(T.eq(test_pred, target_var_sup),
                            dtype=theano.config.floatX) * 100.

        elif disc_nonlinearity == "softmax":
            test_pred = prediction_det.argmax(1)
            test_acc = T.mean(T.eq(test_pred, target_var_sup.argmax(1)),
                            dtype=theano.config.floatX) * 100

        monitor_labels.append("accuracy")
        val_outputs.append(test_acc)

    # Compile prediction function
    predict = theano.function([input_var_sup], test_pred)

    # Compile validation function
    val_fn = theano.function(inputs,
                             [prediction_det] + val_outputs,
                             on_unused_input='ignore')

    # Finally, launch the training loop.
    print("Starting testing...")

    # Load best model
    with np.load(os.path.join(save_copy, 'model_feat_sel.npz')) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values([reconst_net,
                                             discrim_net],
                                            param_values)

    test_minibatches = iterate_minibatches(x_test, y_test, batch_size,
                                           shuffle=False)

    test_err, pred, lab = monitoring(test_minibatches, "test", val_fn,
                                     monitor_labels, prec_recall_cutoff)

    lab = lab.argmax(1)
    pred_argmax = pred.argmax(1)

    cm = np.zeros((26, 26))

    for i in range(26):
        for j in range(26):
            cm[i, j] = ((pred_argmax == i) * (lab == j)).sum()

    np.savez(os.path.join(save_copy, 'cm.npz'), cm)

    print(os.path.join(save_copy, 'cm.npz'))

    # plt.imshow(cm)
    # plt.show()

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


def main():
    parser = argparse.ArgumentParser(description="""Implementation of the
                                     feature selection v2""")
    parser.add_argument('--dataset',
                        default='1000_genomes',
                        help='Dataset.')
    parser.add_argument('--n_hidden_u',
                        default=[100],
                        help='List of unsupervised hidden units.')
    parser.add_argument('--n_hidden_t_enc',
                        default=[100],
                        help='List of theta transformation hidden units.')
    parser.add_argument('--n_hidden_t_dec',
                        default=[100],
                        help='List of theta_prime transformation hidden units')
    parser.add_argument('--n_hidden_s',
                        default=[100],
                        help='List of supervised hidden units.')
    parser.add_argument('--embedding_source',
                        default=None, # 'our_model_aux/feature_embedding.npz',
                        help='Source for the feature embedding. Either' +
                             'None or the name of a file from which' +
                             'to load a learned embedding')
    parser.add_argument('--num_epochs',
                        '-ne',
                        type=int,
                        default=500,
                        help="""Int to indicate the max'
                        'number of epochs.""")
    parser.add_argument('--learning_rate',
                        '-lr',
                        type=float,
                        default=0.00005,
                        help="""Float to indicate learning rate.""")
    parser.add_argument('--learning_rate_annealing',
                        '-lra',
                        type=float,
                        default=1.0,
                        help="Float to indicate learning rate annealing rate.")
    parser.add_argument('--gamma',
                        '-g',
                        type=float,
                        default=0.0,
                        help="""reconst_loss coeff.""")
    parser.add_argument('--disc_nonlinearity',
                        '-nl',
                        default="softmax",
                        help="""Nonlinearity to use in disc_net's last layer""")
    parser.add_argument('--encoder_net_init',
                        '-eni',
                        type=float,
                        default=0.00001,
                        help="Bounds of uniform initialization for " +
                              "encoder_net weights")
    parser.add_argument('--decoder_net_init',
                        '-dni',
                        type=float,
                        default=0.00001,
                        help="Bounds of uniform initialization for " +
                              "decoder_net weights")
    parser.add_argument('--keep_labels',
                        type=float,
                        default=1.0,
                        help='Fraction of training labels to keep')
    parser.add_argument('--prec_recall_cutoff',
                        type=int,
                        help='Whether to compute the precision-recall cutoff' +
                             'or not')
    parser.add_argument('--early_stop_criterion',
                        default='accuracy',
                        help='What monitored variable to use for early-stopping')
    parser.add_argument('--save_perm',
                        default='/data/lisatmp4/'+ os.environ["USER"]+'/feature_selection/',
                        help='Path to save results.')
    parser.add_argument('--dataset_path',
                        default='/data/lisatmp4/romerosa/datasets/',
                        help='Path to dataset')

    args = parser.parse_args()
    print ("Printing args")
    print (args)

    execute(args.dataset,
            parse_int_list_arg(args.n_hidden_u),
            parse_int_list_arg(args.n_hidden_t_enc),
            parse_int_list_arg(args.n_hidden_t_dec),
            parse_int_list_arg(args.n_hidden_s),
            args.embedding_source,
            int(args.num_epochs),
            args.learning_rate,
            args.learning_rate_annealing,
            args.gamma,
            args.disc_nonlinearity,
            args.encoder_net_init,
            args.decoder_net_init,
            args.keep_labels,
            args.prec_recall_cutoff != 0, -1,
            args.early_stop_criterion,
            args.save_perm,
            args.dataset_path)


if __name__ == '__main__':
    main()
