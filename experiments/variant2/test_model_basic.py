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
from lasagne.regularization import apply_penalty, l2

import numpy as np
import theano
import theano.tensor as T

from DietNetworks.experiments.common import dataset_utils

import mainloop_helpers as mlh
import model_helpers as mh

import getpass
CLUSTER = getpass.getuser() in ["tisu32"]


# Main program
def execute(dataset, n_hidden_t_enc, n_hidden_s,
            num_epochs=500, learning_rate=.001, learning_rate_annealing=1.0,
            gamma=1, lmd=0., disc_nonlinearity="sigmoid", keep_labels=1.0,
            prec_recall_cutoff=True, missing_labels_val=-1.0,  which_fold=1,
            early_stop_criterion='loss', dataset_path='/Tmp/carriepl/datasets/',
            save_path='/data/lisatmp4/romerosa/DietNetworks/',
            resume=False, exp_name=None):

    # Load the dataset
    print("Loading data")
    x_train, y_train, x_valid, y_valid, x_test, y_test, \
        x_unsup, training_labels = mlh.load_data(
            dataset, dataset_path, None,
            which_fold=which_fold, keep_labels=keep_labels,
            missing_labels_val=missing_labels_val,
            embedding_input='raw')

    # Extract required information from data
    n_samples, n_feats = x_train.shape
    print("Number of features : ", n_feats)
    print("Glorot init : ", 2.0 / (n_feats + n_hidden_t_enc[-1]))
    n_targets = y_train.shape[1]

    # Set some variables
    batch_size = 138

    # Preparing folder to save stuff
    exp_name = 'basic_' + mlh.define_exp_name(keep_labels, 0, 0, gamma, lmd,
                                              [], n_hidden_t_enc, [],
                                              n_hidden_s, which_fold,
                                              learning_rate, 0,
                                              0, early_stop_criterion,
                                              learning_rate_annealing)

    print("Experiment: " + exp_name)
    save_path = os.path.join(save_path, dataset, exp_name)
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    # Prepare Theano variables for inputs and targets
    input_var_sup = T.matrix('input_sup')
    target_var_sup = T.matrix('target_sup')
    lr = theano.shared(np.float32(learning_rate), 'learning_rate')

    # Build model
    print("Building model")
    discrim_net = InputLayer((None, n_feats), input_var_sup)
    discrim_net = DenseLayer(discrim_net, num_units=n_hidden_t_enc[-1],
                             nonlinearity=rectify)

    # Reconstruct the input using dec_feat_emb
    if gamma > 0:
        reconst_net = DenseLayer(discrim_net, num_units=n_feats,
                                 nonlinearity=linear)
        nets = [reconst_net]
    else:
        nets = [None]

    # Add supervised hidden layers
    for hid in n_hidden_s:
        discrim_net = DropoutLayer(discrim_net)
        discrim_net = DenseLayer(discrim_net, num_units=hid)

    assert disc_nonlinearity in ["sigmoid", "linear", "rectify", "softmax"]
    discrim_net = DropoutLayer(discrim_net)
    discrim_net = DenseLayer(discrim_net, num_units=n_targets,
                             nonlinearity=eval(disc_nonlinearity))

    # Load best model
    with np.load(os.path.join(save_path, 'model_best.npz')) as f:
        param_values = [f['arr_%d' % i]
                        for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(filter(None, nets) +
                                        [discrim_net],
                                        param_values)

    print("Building and compiling training functions")

    # Build and compile functions
    predictions, predictions_det = mh.define_predictions(nets, start=0)
    prediction_sup, prediction_sup_det = mh.define_predictions([discrim_net])
    prediction_sup = prediction_sup[0]
    prediction_sup_det = prediction_sup_det[0]

    # Define losses
    # reconstruction losses
    _, reconst_losses_det = mh.define_reconst_losses(
        predictions, predictions_det, [input_var_sup])
    # supervised loss
    _, sup_loss_det = mh.define_sup_loss(
        disc_nonlinearity, prediction_sup, prediction_sup_det, keep_labels,
        target_var_sup, missing_labels_val)

    inputs = [input_var_sup, target_var_sup]

    params = lasagne.layers.get_all_params(
        [discrim_net] + filter(None, nets), trainable=True)

    # Combine losses
    loss_det = sup_loss_det + gamma*reconst_losses_det[0]

    l2_penalty = apply_penalty(params, l2)
    loss_det = loss_det + lmd*l2_penalty

    # Monitoring Labels
    monitor_labels = ["reconst. loss"]
    monitor_labels = [i for i, j in zip(monitor_labels, reconst_losses_det)
                      if j != 0]
    monitor_labels += ["loss. sup.", "total loss"]

    # Build and compile test function
    val_outputs = reconst_losses_det
    val_outputs = [i for i, j in zip(val_outputs, reconst_losses_det) if j != 0]
    val_outputs += [sup_loss_det, loss_det]

    # Compute accuracy and add it to monitoring list
    test_acc, test_pred = mh.define_test_functions(
        disc_nonlinearity, prediction_sup, prediction_sup_det, target_var_sup)
    monitor_labels.append("accuracy")
    val_outputs.append(test_acc)

    # Compile prediction function
    predict = theano.function([input_var_sup], test_pred)

    # Compile validation function
    val_fn = theano.function(inputs,
                             [prediction_sup_det] + val_outputs,
                             on_unused_input='ignore')

    # Finally, launch the training loop.
    print("Starting testing...")
    test_minibatches = mlh.iterate_minibatches(x_test, y_test,
                                               batch_size,
                                               shuffle=False)

    test_err, pred, targets = mlh.monitoring(test_minibatches, "test", val_fn,
                                             monitor_labels, prec_recall_cutoff,
                                             return_pred=True)

    lab = targets.argmax(1)
    pred_argmax = pred.argmax(1)

    continent_cat = mh.create_1000_genomes_continent_labels()
    lab_cont = np.zeros(lab.shape)
    pred_cont = np.zeros(pred_argmax.shape)

    for i,c in enumerate(continent_cat):
        for el in c:
            lab_cont[lab == el] = i
            pred_cont[pred_argmax == el] = i

    cm_e = np.zeros((26, 26))
    cm_c = np.zeros((5,5))

    for i in range(26):
        for j in range(26):
            cm_e[i, j] = ((pred_argmax == i) * (lab == j)).sum()

    for i in range(5):
        for j in range(5):
            cm_c[i, j] = ((pred_cont == i) * (lab_cont == j)).sum()

    np.savez(os.path.join(save_path, 'cm'+str(which_fold)+'.npz'),
             cm_e=cm_e, cm_c=cm_c)

    print(os.path.join(save_path, 'cm' + str(which_fold) + '.npz'))

def main():
    parser = argparse.ArgumentParser(description="""Test basic model""")
    parser.add_argument('--dataset',
                        default='1000_genomes',
                        help='Dataset.')
    parser.add_argument('--n_hidden_t_enc',
                        default=[100],
                        help='List of theta transformation hidden units.')
    parser.add_argument('--n_hidden_s',
                        default=[100],
                        help='List of supervised hidden units.')
    parser.add_argument('--num_epochs',
                        '-ne',
                        type=int,
                        default=1000,
                        help='Int to indicate the max number of epochs.')
    parser.add_argument('--learning_rate',
                        '-lr',
                        type=float,
                        default=0.0001,
                        help='Float to indicate learning rate.')
    parser.add_argument('--learning_rate_annealing',
                        '-lra',
                        type=float,
                        default=.99,
                        help='Float to indicate learning rate annealing rate.')
    parser.add_argument('--gamma',
                        '-g',
                        type=float,
                        default=0.,
                        help='reconst_loss coeff.')
    parser.add_argument('--lmd',
                        '-l',
                        type=float,
                        default=0.,
                        help="""Weight decay coeff.""")
    parser.add_argument('--disc_nonlinearity',
                        '-nl',
                        default="softmax",
                        help='Nonlinearity to use in disc_net last layer')
    parser.add_argument('--keep_labels',
                        type=float,
                        default=1.0,
                        help='Fraction of training labels to keep')
    parser.add_argument('--prec_recall_cutoff',
                        type=int,
                        help='Whether to compute the precision-recall cutoff' +
                             'or not')
    parser.add_argument('--which_fold',
                        type=int,
                        default=1,
                        help='Which fold to use for cross-validation (0-4)')
    parser.add_argument('--early_stop_criterion',
                        default='accuracy',
                        help='What monitored variable to use for early-stopping')
    parser.add_argument('--dataset_path',
                        default='/data/lisatmp4/romerosa/datasets/1000_Genome_project/',
                        help='Path to dataset')
    parser.add_argument('--save_path',
                        default='/data/lisatmp4/romerosa/DietNetworks/',
                        help='Path to save results.')
    parser.add_argument('-resume',
                        type=bool,
                        default=False,
                        help='Whether to resume job')
    parser.add_argument('-exp_name',
                        type=str,
                        default='',
                        help='Experiment name that will be concatenated at the beginning of the generated name')

    args = parser.parse_args()
    print ("Printing args")
    print (args)

    execute(args.dataset,
            mlh.parse_int_list_arg(args.n_hidden_t_enc),
            mlh.parse_int_list_arg(args.n_hidden_s),
            int(args.num_epochs),
            args.learning_rate,
            args.learning_rate_annealing,
            args.gamma,
            args.lmd,
            args.disc_nonlinearity,
            args.keep_labels,
            args.prec_recall_cutoff != 0, -1,
            args.which_fold,
            args.early_stop_criterion,
            args.dataset_path,
            args.save_path,
            args.resume,
            args.exp_name)


if __name__ == '__main__':
    main()
