#!/usr/bin/env python


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
from theano import config
import theano.tensor as T

from feature_selection.experiments.common import dataset_utils, imdb

print ("config floatX: {}".format(config.floatX))


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


# Main program
def execute(dataset, n_hidden_u, n_hidden_t_enc, n_hidden_t_dec, n_hidden_s,
            embedding_source=None,
            num_epochs=500, learning_rate=.001, learning_rate_annealing=1.0,
            gamma=1, disc_nonlinearity="sigmoid", encoder_net_init=0.2,
            decoder_net_init=0.2, keep_labels=1.0, prec_recall_cutoff=True,
            missing_labels_val=-1.0, which_fold=0, early_stop_criterion='loss_sup_det',
            save_path='/Tmp/romerosa/feature_selection/newmodel/',
            save_copy='/Tmp/romerosa/feature_selection/',
            dataset_path='/Tmp/' + os.environ["USER"] + '/datasets/'):

    # Load the dataset
    print("Loading data")
    # This will split the training data into 60% train, 20% valid, 20% test
    splits = [.6, .2]

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
        data = dataset_utils.load_iric_molecules(
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
        # import ipdb; ipdb.set_trace()
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
        from feature_selection.experiments.common import dragonn_data
        data = dragonn_data.load_data(500, 100, 100)
    elif dataset == '1000_genomes':
        # This will split the training data into 75% train, 25%
        # this corresponds to the split 60/20 of the whole data,
        # test is considered elsewhere as an extra 20% of the whole data
        splits = [.75]
        data = dataset_utils.load_1000_genomes(transpose=False,
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
        n_samples_unsup = x_unsup.shape[1]
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

    # Extract required information from data
    n_samples, n_feats = x_train.shape
    print("Number of features : ", n_feats)
    print("Glorot init : ", 2.0 / (n_feats + n_hidden_t_enc[-1]))
    n_targets = y_train.shape[1]

    # Set some variables
    batch_size = 128

    # Preparing folder to save stuff
    exp_name = 'our_model' + str(keep_labels) + '_sup' + \
        ('_unsup' if gamma > 0 else '')
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
    print("Experiment: " + exp_name)
    save_path = os.path.join(save_path, dataset, exp_name)
    save_copy = os.path.join(save_copy, dataset, exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

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
    if not embedding_source:
        encoder_net = InputLayer((n_feats, n_samples_unsup), input_var_unsup)
        for i, out in enumerate(n_hidden_u):
            encoder_net = DenseLayer(encoder_net, num_units=out,
                                     nonlinearity=rectify)
        feat_emb = lasagne.layers.get_output(encoder_net)
        pred_feat_emb = theano.function([], feat_emb)
    else:
        feat_emb_val = np.load(os.path.join(save_path.rsplit('/', 1)[0],
                                            embedding_source)).items()[0][1]
        feat_emb = theano.shared(feat_emb_val, 'feat_emb')
        encoder_net = InputLayer((n_feats, n_hidden_u[-1]), feat_emb)

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
    # Some variables
    loss_sup = 0
    loss_sup_det = 0

    # Build and compile training functions

    prediction = lasagne.layers.get_output(discrim_net)
    prediction_det = lasagne.layers.get_output(discrim_net,
                                               deterministic=True)

    # Supervised loss
    if disc_nonlinearity == "sigmoid":
        loss_sup = lasagne.objectives.binary_crossentropy(
            prediction, target_var_sup)
        loss_sup_det = lasagne.objectives.binary_crossentropy(
            prediction_det, target_var_sup)
    elif disc_nonlinearity == "softmax":
        loss_sup = lasagne.objectives.categorical_crossentropy(prediction,
                                                               target_var_sup)
        loss_sup_det = lasagne.objectives.categorical_crossentropy(prediction_det,
                                                                   target_var_sup)
    elif disc_nonlinearity in ["linear", "rectify"]:
        loss_sup = lasagne.objectives.squared_error(
            prediction, target_var_sup)
        loss_sup_det = lasagne.objectives.squared_error(
            prediction_det, target_var_sup)
    else:
        raise ValueError("Unsupported non-linearity")

    # If some labels are missing, mask the appropriate losses before taking
    # the mean.
    if keep_labels < 1.0:
        mask = T.neq(target_var_sup, missing_labels_val)
        scale_factor = 1.0 / mask.mean()
        loss_sup = (loss_sup * mask) * scale_factor
        loss_sup_det = (loss_sup_det * mask) * scale_factor
    loss_sup = loss_sup.mean()
    loss_sup_det = loss_sup_det.mean()

    inputs = [input_var_sup, target_var_sup]

    # Unsupervised reconstruction loss
    reconstruction = lasagne.layers.get_output(reconst_net)
    reconstruction_det = lasagne.layers.get_output(reconst_net,
                                                   deterministic=True)
    reconst_loss = lasagne.objectives.squared_error(
        reconstruction,
        input_var_sup).mean()
    reconst_loss_det = lasagne.objectives.squared_error(
        reconstruction_det,
        input_var_sup).mean()

    params = lasagne.layers.get_all_params([discrim_net, reconst_net,
                                            encoder_net_W_dec,
                                            encoder_net_W_enc],
                                           trainable=True)

    # Combine losses
    loss = loss_sup + gamma*reconst_loss
    loss_det = loss_sup_det + gamma*reconst_loss_det

    # Compute network updates
    updates = lasagne.updates.rmsprop(loss,
                                      params,
                                      learning_rate=lr)
    # updates = lasagne.updates.sgd(loss,
    #                               params,
    #                               learning_rate=lr)
    # updates = lasagne.updates.momentum(loss, params,
    #                                    learning_rate=lr, momentum=0.0)

    # Apply norm constraints on the weights
    for k in updates.keys():
        if updates[k].ndim == 2:
            updates[k] = lasagne.updates.norm_constraint(updates[k], 1.0)

    # Compile training function
    train_fn = theano.function(inputs, loss, updates=updates,
                               on_unused_input='ignore')

    # Expressions required for test
    monitor_labels = ["total_loss_det", "loss_sup_det", "recon. loss",
                      "enc_w_mean", "enc_w_var"]
    val_outputs = [loss_det, loss_sup_det, reconst_loss_det,
                   enc_feat_emb.mean(), enc_feat_emb.var()]

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
    print("Starting training...")

    # Some variables
    max_patience = 100
    patience = 0

    train_monitored = []
    valid_monitored = []
    train_loss = []

    # Pre-training monitoring
    print("Epoch 0 of {}".format(num_epochs))

    train_minibatches = iterate_minibatches(x_train, y_train,
                                            batch_size, shuffle=False)

    train_errr = monitoring(train_minibatches, "train", val_fn, monitor_labels,
                            prec_recall_cutoff)

    valid_minibatches = iterate_minibatches(x_valid, y_valid,
                                            batch_size, shuffle=False)
    valid_err = monitoring(valid_minibatches, "valid", val_fn, monitor_labels,
                           prec_recall_cutoff)

    # Training loop
    start_training = time.time()
    for epoch in range(num_epochs):
        start_time = time.time()
        print("Epoch {} of {}".format(epoch+1, num_epochs))
        nb_minibatches = 0
        loss_epoch = 0

        # Train pass
        for batch in iterate_minibatches(x_train, training_labels,
                                         batch_size,
                                         shuffle=True):
            loss_epoch += train_fn(*batch)
            nb_minibatches += 1

        loss_epoch /= nb_minibatches
        train_loss += [loss_epoch]

        # Monitoring on the training set
        train_minibatches = iterate_minibatches(x_train, y_train,
                                                batch_size, shuffle=False)
        train_err = monitoring(train_minibatches, "train", val_fn,
                               monitor_labels, prec_recall_cutoff)
        train_monitored += [train_err]

        # Monitoring on the validation set
        valid_minibatches = iterate_minibatches(x_valid, y_valid,
                                                batch_size, shuffle=False)

        valid_err = monitoring(valid_minibatches, "valid", val_fn,
                               monitor_labels, prec_recall_cutoff)
        valid_monitored += [valid_err]

        # Monitoring on the test set
        if y_test is not None:
            test_minibatches = iterate_minibatches(x_test, y_test, batch_size,
                                                   shuffle=False)
            test_err = monitoring(test_minibatches, "test", val_fn,
                                  monitor_labels, prec_recall_cutoff)

        try:
            early_stop_val = valid_err[monitor_labels.index(early_stop_criterion)]
        except:
            raise ValueError("There is no monitored value by the name of %s" %
                             early_stop_criterion)

        # Early stopping
        if epoch == 0:
            best_valid = early_stop_val
        elif early_stop_val > best_valid: # be careful with that!!
            best_valid = early_stop_val
            patience = 0

            # Save stuff
            np.savez(os.path.join(save_path, 'model_feat_sel.npz'),
                     *lasagne.layers.get_all_param_values([reconst_net,
                                                           discrim_net]))
            np.savez(save_path + "errors_supervised.npz",
                     zip(*train_monitored), zip(*valid_monitored))
        else:
            patience += 1

        # End training
        if patience == max_patience or epoch == num_epochs-1:
            print("Ending training")
            # print "Patience %i" % (patience)
            # print "Max patience %i" % (max_patience)
            # print "Epoch %i" % (epoch)
            # print "Num epochs %i" % (num_epochs)

            # Load best model
            with np.load(os.path.join(save_path, 'model_feat_sel.npz')) as f:
                param_values = [f['arr_%d' % i]
                                for i in range(len(f.files))]
            nlayers = len(lasagne.layers.get_all_params([reconst_net,
                                                        discrim_net]))
            lasagne.layers.set_all_param_values([reconst_net,
                                                discrim_net],
                                                param_values[:nlayers])
            if embedding_source is None:
                # Save embedding
                pred = pred_feat_emb()
                np.savez(os.path.join(save_path, 'feature_embedding.npz'), pred)

            # Test
            if y_test is not None:
                test_minibatches = iterate_minibatches(x_test, y_test,
                                                       batch_size,
                                                       shuffle=False)

                test_err = monitoring(test_minibatches, "test", val_fn,
                                      monitor_labels, prec_recall_cutoff)
            else:
                for minibatch in iterate_testbatches(x_test,
                                                     batch_size,
                                                     shuffle=False):
                    test_predictions = []
                    test_predictions += [predict(minibatch)]
                np.savez(os.path.join(save_path, 'test_predictions.npz'),
                                      test_predictions)

            train_minibatches = iterate_minibatches(x_train, y_train,
                                                    batch_size,
                                                    shuffle=False)
            train_err = monitoring(train_minibatches, "train", val_fn,
                                   monitor_labels, prec_recall_cutoff)

            valid_minibatches = iterate_minibatches(x_valid, y_valid,
                                                    batch_size,
                                                    shuffle=False)
            valid_err = monitoring(valid_minibatches, "valid", val_fn,
                                   monitor_labels, prec_recall_cutoff)
            # Stop
            print("  epoch time:\t\t\t{:.3f}s \n".format(time.time() -
                                                         start_time))
            break

        print("  epoch time:\t\t\t{:.3f}s \n".format(time.time() - start_time))

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
                                     feature selection v2""")
    parser.add_argument('--dataset',
                        default='1000_genomes',
                        help='Dataset.')
    parser.add_argument('--n_hidden_u',
                        default=[30],
                        help='List of unsupervised hidden units.')
    parser.add_argument('--n_hidden_t_enc',
                        default=[30],
                        help='List of theta transformation hidden units.')
    parser.add_argument('--n_hidden_t_dec',
                        default=[30],
                        help='List of theta_prime transformation hidden units')
    parser.add_argument('--n_hidden_s',
                        default=[30],
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
                        default=0.00001,
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
                        default=0,
                        help='Whether to compute the precision-recall cutoff' +
                             'or not')
    parser.add_argument('--which_fold',
                        type=int,
                        default=0,
                        help='Which fold to use for cross-validation (0-4)')
    parser.add_argument('--early_stop_criterion',
                        default='accuracy',
                        help='What monitored variable to use for early-stopping')
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
            args.which_fold,
            args.early_stop_criterion,
            args.save_tmp,
            args.save_perm,
            args.dataset_path)


if __name__ == '__main__':
    main()
