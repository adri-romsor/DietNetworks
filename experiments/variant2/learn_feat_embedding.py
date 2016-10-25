#!/usr/bin/env python2

from __future__ import print_function
import argparse
import time
import os
import tables
from distutils.dir_util import copy_tree

import lasagne
from lasagne.layers import DenseLayer, InputLayer, DropoutLayer
from lasagne.nonlinearities import sigmoid, softmax, tanh, linear, rectify
from lasagne.regularization import apply_penalty, l2, l1
from lasagne.init import Uniform
import numpy as np
import theano
import theano.tensor as T

from epls import EPLS, tensor_fun_EPLS
from feature_selection.experiments.common import dataset_utils, imdb


def iterate_minibatches(x, batch_size, shuffle=False, dataset=None):
    indices = np.arange(x.shape[0])
    if shuffle:
        indices = np.random.permutation(x.shape[0])
    for i in range(0, x.shape[0]-batch_size+1, batch_size):
        yield x[indices[i:i+batch_size], :]


def monitoring(minibatches, which_set, error_fn, monitoring_labels):
    print('-'*20 + which_set + ' monit.' + '-'*20)
    monitoring_values = np.zeros(len(monitoring_labels), dtype="float32")
    global_batches = 0

    for batch in minibatches:
        # Update monitored values
        out = error_fn(batch)

        monitoring_values = monitoring_values + out
        global_batches += 1

    # Print monitored values
    monitoring_values /= global_batches
    for (label, val) in zip(monitoring_labels, monitoring_values):
        print ("  {} {}:\t\t{:.6f}".format(which_set, label, val))

    return monitoring_values


# Main program
def execute(dataset, n_hidden_u, unsupervised=[], num_epochs=500,
            learning_rate=.001, learning_rate_annealing=1.0, lmd=.0001,
            embedding_input='raw',
            save_path='/Tmp/$USER/feature_selection/newmodel/',
            save_copy='/Tmp/$USER/feature_selection/newmodel/',
            dataset_path='/Tmp/$USER/feature_selection/newmodel/'):

    # Load the dataset
    print("Loading data")
    splits = [0.80]  # This will split the data into [80%, 20%]
    if dataset == 'protein_binding':
        data = dataset_utils.load_protein_binding(transpose=True,
                                                  splits=splits)
    elif dataset == 'dorothea':
        data = dataset_utils.load_dorothea(transpose=True, splits=splits)
    elif dataset == 'opensnp':
        data = dataset_utils.load_opensnp(transpose=True, splits=splits)
    elif dataset == 'reuters':
        data = dataset_utils.load_reuters(transpose=True, splits=splits)
    elif dataset == 'imdb':
        # data = dataset_utils.load_imdb(transpose=True, splits=splits)
        # train_data, _, unlab_data, _ = imdb.load_imdb(feat_type=feat_type,
        #                                               ngram_range=ngram_range)
        data = imdb.read_from_hdf5(unsupervised=True, feat_type='tfidf')
    elif dataset == 'dragonn':
        from feature_selection.experiments.common import dragonn_data
        data = dragonn_data.load_data(500, 10000, 10000)
    elif dataset == '1000_genomes':
        data = dataset_utils.load_1000_genomes(
                   transpose=True, label_splits=splits, feature_splits=splits,
                   nolabels=embedding_input)
    else:
        print("Unknown dataset")
        return

    if dataset == 'imdb':
        x_train = data.root.train
        x_valid = data.root.val
    else:
        x_train = data[0][0]
        x_valid = data[1][0]

    # Extract required information from data
    n_row, n_col = x_train.shape

    # Set some variables
    batch_size = 256

    # Preparing folder to save stuff
    exp_name = 'our_model_aux_glorot_' + str(learning_rate)
    exp_name += '_hu'
    for i in range(len(n_hidden_u)):
        exp_name += ("-" + str(n_hidden_u[i]))
    print('Experiment: ' + exp_name)
    print('Data size ' + str(n_row) + 'x' + str(n_col))

    save_path = os.path.join(save_path, dataset, exp_name)
    save_copy = os.path.join(save_copy, dataset, exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('input_unsup')
    lr = theano.shared(np.float32(learning_rate), 'learning_rate')
    update_epls = {}

    # Build model
    print("Building model")

    # Some checkings
    assert len(n_hidden_u) > 0

    # Build unsupervised network
    encoder_net = InputLayer((None, n_col), input_var)

    for out in n_hidden_u:
        encoder_net = DenseLayer(encoder_net, num_units=out,
                                 nonlinearity=tanh)
        encoder_net = DropoutLayer(encoder_net)
    feat_emb = lasagne.layers.get_output(encoder_net)
    pred_feat_emb = theano.function([input_var], feat_emb)

    if 'autoencoder' in unsupervised:
        decoder_net = encoder_net
        for i in range(len(n_hidden_u)-2, -1, -1):
            decoder_net = DenseLayer(decoder_net, num_units=n_hidden_u[i],
                                     nonlinearity=tanh)
            decoder_net = DropoutLayer(decoder_net)
        decoder_net = DenseLayer(decoder_net, num_units=n_col,
                                 nonlinearity=linear)
        reconstruction = lasagne.layers.get_output(decoder_net)
    if 'epls' in unsupervised:
        n_cluster = n_hidden_u[-1]
        activation = theano.shared(np.zeros(n_cluster, dtype='float32'))
        nb_activation = 1

        # /!\ if non linearity not sigmoid, map output values to function image
        hidden_unsup = T.nnet.sigmoid(feat_emb)

        target, new_act = tensor_fun_EPLS(hidden_unsup, activation,
                                          n_row, nb_activation)

        update_epls[activation] = new_act
        # h_rep = T.largest(0, h_rep - T.mean(h_rep))

    print("Building and compiling training functions")
    # Some variables
    loss_auto = 0
    loss_auto_det = 0
    loss_epls = 0
    loss_epls_det = 0
    params = []

    # Build and compile training functions
    if "autoencoder" in unsupervised:
        # Unsupervised reconstruction functions
        reconstruction = lasagne.layers.get_output(decoder_net)
        reconstruction_det = lasagne.layers.get_output(decoder_net,
                                                       deterministic=True)

        loss_auto = lasagne.objectives.squared_error(
            reconstruction,
            input_var).mean()
        loss_auto_det = lasagne.objectives.squared_error(
            reconstruction_det,
            input_var).mean()

        params += lasagne.layers.get_all_params(decoder_net, trainable=True)
    if "epls" in unsupervised:
        # Unsupervised epls functions
        loss_epls = ((hidden_unsup - target) ** 2).mean()
        loss_epls_det = ((hidden_unsup - target) ** 2).mean()

    # Combine losses
    loss = loss_auto + loss_epls
    loss_det = loss_auto_det + loss_epls_det

    l2_penalty = apply_penalty(params, l2)
    loss = loss + lmd*l2_penalty
    loss_det = loss_det + lmd*l2_penalty

    # Compute network updates
    updates = lasagne.updates.adam(loss,
                                   params,
                                   learning_rate=lr)
    # updates = lasagne.updates.sgd(loss,
    #                              params,
    #                              learning_rate=lr)
    # updates = lasagne.updates.momentum(loss, params,
    #                                    learning_rate=lr, momentum=0.0)

    if 'epls' in unsupervised:
        updates[activation] = new_act

    # Compile training function
    train_fn = theano.function([input_var], loss,
                               updates=updates,
                               on_unused_input='ignore')

    # Expressions required for test
    monitor_labels = ['total_loss_det']
    val_outputs = [loss_det]

    if "autoencoder" in unsupervised:
        # Unsupervised reconstruction functions
        val_outputs += [loss_auto_det]
        monitor_labels += ["recon. loss"]

    if "epls" in unsupervised:
        # Unsupervised epls functions
        val_outputs += [loss_epls_det]
        monitor_labels += ["epls. loss"]

    # Add some monitoring on the learned feature embedding
    val_outputs += [feat_emb.min(), feat_emb.mean(),
                    feat_emb.max(), feat_emb.var()]
    monitor_labels += ["feat. emb. min", "feat. emb. mean",
                       "feat. emb. max", "feat. emb. var"]

    # Compile validation function
    val_fn = theano.function([input_var],
                             val_outputs,
                             updates=update_epls)

    # Finally, launch the training loop.
    print("Starting training...")

    # Some variables
    max_patience = 100
    patience = 0

    train_loss = []
    train_loss_auto = []
    train_loss_epls = []
    valid_loss = []
    valid_loss_auto = []
    valid_loss_epls = []

    nb_minibatches = n_row/batch_size
    print("Nb of minibatches: " + str(nb_minibatches))
    start_training = time.time()
    for epoch in range(num_epochs):
        start_time = time.time()
        print("Epoch {} of {}".format(epoch+1, num_epochs))

        loss_epoch = 0

        # Train pass
        for batch in iterate_minibatches(x_train, batch_size,
                                         dataset=dataset, shuffle=True):
            loss_epoch += train_fn(batch)

        train_minibatches = iterate_minibatches(x_train, batch_size,
                                                dataset=dataset, shuffle=True)
        train_err = monitoring(train_minibatches, "train", val_fn,
                               monitor_labels)
        train_loss += [train_err[0]]
        pos = 1
        if 'autoencoder' in unsupervised:
            train_loss_auto += [train_err[pos]]
            pos += 1
        if 'epls' in unsupervised:
            train_loss_epls += [train_err[pos]]

        # Validation pass
        valid_minibatches = iterate_minibatches(x_valid, batch_size,
                                                dataset=dataset, shuffle=True)

        valid_err = monitoring(valid_minibatches, "valid", val_fn,
                               monitor_labels)

        valid_loss += [valid_err[0]]
        pos = 1
        if 'autoencoder' in unsupervised:
            valid_loss_auto += [valid_err[pos]]
            pos += 1
        if 'epls' in unsupervised:
            valid_loss_epls += [valid_err[pos]]

        # Eearly stopping
        if epoch == 0:
            best_valid = valid_loss[epoch]
        elif valid_loss[epoch] < best_valid:
            best_valid = valid_loss[epoch]
            patience = 0

            # Save stuff
            np.savez(os.path.join(save_path, 'model_unsupervised.npz'),
                     *lasagne.layers.get_all_param_values(encoder_net))
            np.savez(os.path.join(save_path, "errors_unsupervised.npz"),
                     train_loss, train_loss_auto, train_loss_epls,
                     valid_loss, valid_loss_auto, valid_loss_epls)
        else:
            patience += 1

        # End training
        if patience == max_patience or epoch == num_epochs-1:
            print("   Ending training")
            # Load unsupervised best model
            with np.load(os.path.join(save_path,
                                      'model_unsupervised.npz')) as f:
                param_values = [f['arr_%d' % i]
                                for i in range(len(f.files))]
                nlayers = len(lasagne.layers.get_all_params(encoder_net))
                lasagne.layers.set_all_param_values(encoder_net,
                                                    param_values[:nlayers])

                # Save embedding
                preds = []
                for batch in iterate_minibatches(x_train, 1, dataset=dataset,
                                                 shuffle=False):
                    preds.append(pred_feat_emb(batch))
                for batch in iterate_minibatches(x_valid, 1, dataset=dataset,
                                                 shuffle=False):
                    preds.append(pred_feat_emb(batch))
                preds = np.vstack(preds)
                np.savez(os.path.join(save_path, 'feature_embedding.npz'),
                         preds)

            # Stop
            print(" epoch time:\t\t\t{:.3f}s".format(time.time() - start_time))
            break

        print("  epoch time:\t\t\t{:.3f}s".format(time.time() - start_time))
        # Anneal the learning rate
        lr.set_value(float(lr.get_value() * learning_rate_annealing))

    # Print all final errors for train, validation and test
    print("Training time:\t\t\t{:.3f}s".format(time.time() - start_training))

    # Copy files to loadpath
    if save_path != save_copy:
        print('Copying model and other training files to {}'.format(save_copy))
        copy_tree(save_path, save_copy)


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
                                     feature selection v4""")
    parser.add_argument('--dataset',
                        default='1000_genomes',
                        help='Dataset.')
    parser.add_argument('--n_hidden_u',
                        default=[100],
                        help='List of unsupervised hidden units.')
    parser.add_argument('--unsupervised',
                        default=['autoencoder'],
                        help='Add unsupervised part of the network:' +
                             'list containinge autoencoder and/or epls' +
                             'or []')
    parser.add_argument('--num_epochs',
                        '-ne',
                        type=int,
                        default=1000,
                        help="""Int to indicate the max'
                        'number of epochs.""")
    parser.add_argument('--learning_rate',
                        '-lr',
                        type=float,
                        default=.05,
                        help="""Float to indicate learning rate.""")
    parser.add_argument('--learning_rate_annealing',
                        '-lra',
                        type=float,
                        default=.99,
                        help="Float to indicate learning rate annealing rate.")
    parser.add_argument('--lmd',
                        '-l',
                        type=float,
                        default=.0001,
                        help="""Float to indicate weight decay coeff.""")
    parser.add_argument('-embedding_input',
                        type=str,
                        default='raw',
                        help='The kind of input we will use for the feat. emb. nets')
    parser.add_argument('--save_tmp',
                        default='/Tmp/'+ os.environ["USER"]+'/feature_selection/',
                        help='Path to save results.')
    parser.add_argument('--save_perm',
                        default='/data/lisatmp4/'+ os.environ["USER"]+'/feature_selection/',
                        help='Path to save results.')
    parser.add_argument('--dataset_path',
                        default='/data/lisatmp4/romerosa/datasets/',
                        help='Path to dataset')


    args = parser.parse_args()

    if args.unsupervised == []:
        raise StandardError('you must provide non empty list for unsupervised')
    execute(args.dataset,
            parse_int_list_arg(args.n_hidden_u),
            args.unsupervised,
            int(args.num_epochs),
            args.learning_rate,
            args.learning_rate_annealing,
            args.lmd,
            args.embedding_input,
            args.save_tmp,
            args.save_perm,
            args.dataset_path)


if __name__ == '__main__':
    main()
