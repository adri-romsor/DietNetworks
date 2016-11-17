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

from lasagne.init import Uniform
import numpy as np
import theano
import theano.tensor as T

from feature_selection.experiments.common import dataset_utils

import matplotlib.pyplot as plt

import mainloop_helpers as mlh
import model_helpers as mh


# Main program
def execute(dataset, n_hidden_u, n_hidden_t_enc, n_hidden_t_dec, n_hidden_s,
            embedding_source=None,
            num_epochs=500, learning_rate=.001, learning_rate_annealing=1.0,
            alpha=1, beta=1, gamma=1, lmd=.0001, disc_nonlinearity="sigmoid",
            encoder_net_init=0.2, decoder_net_init=0.2, keep_labels=1.0,
            prec_recall_cutoff=True, missing_labels_val=-1.0, which_fold=0,
            early_stop_criterion='loss_sup_det', embedding_input='raw',
            model_path='/Tmp/romerosa/feature_selection/newmodel/',
            save_path='/Tmp/romerosa/feature_selection/',
            dataset_path='/Tmp/' + os.environ["USER"] + '/datasets/',
            resume=False, exp_name=''):

    # Load the dataset
    print("Loading data")
    x_train, y_train, x_valid, y_valid, x_test, y_test, \
        x_unsup, training_labels = mlh.load_data(
            dataset, dataset_path, embedding_source,
            which_fold=which_fold, keep_labels=keep_labels,
            missing_labels_val=missing_labels_val,
            embedding_input=embedding_input)

    if x_unsup is not None:
        n_samples_unsup = x_unsup.shape[1]
    else:
        n_samples_unsup = 0

    # Extract required information from data
    n_samples, n_feats = x_train.shape
    print("Number of features : ", n_feats)
    print("Glorot init : ", 2.0 / (n_feats + n_hidden_t_enc[-1]))
    n_targets = y_train.shape[1]

    # Set some variables
    batch_size = 1
    beta = gamma if (gamma == 0) else beta

    # Preparing folder to save stuff
    if embedding_source is None:
        embedding_name = embedding_input
    else:
        embedding_name = embedding_source.replace("_", "").split(".")[0]
        # exp_name = embedding_name.rsplit('/', 1)[::-1][0] + '_'

    # exp_name += '_new_'

    # exp_name += mlh.define_exp_name(keep_labels, alpha, beta, gamma, lmd,
    #                                n_hidden_u, n_hidden_t_enc, n_hidden_t_dec,
    #                                n_hidden_s, which_fold, embedding_input,
    #                                 learning_rate, decoder_net_init,
    #                                 encoder_net_init, early_stop_criterion,
    #                                 learning_rate_annealing)

    print("Experiment: " + exp_name)
    model_path = os.path.join(model_path, dataset, exp_name)
    print(model_path)
    save_path = os.path.join(save_path, dataset, exp_name)
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
    # assert len(n_hidden_u) > 0
    assert len(n_hidden_t_enc) > 0
    assert len(n_hidden_t_dec) > 0
    assert n_hidden_t_dec[-1] == n_hidden_t_enc[-1]

    # Build feature embedding networks (encoding and decoding if gamma > 0)
    nets, embeddings, pred_feat_emb = mh.build_feat_emb_nets(
        embedding_source, n_feats, n_samples_unsup,
        input_var_unsup, n_hidden_u, n_hidden_t_enc,
        n_hidden_t_dec, gamma, encoder_net_init,
        decoder_net_init, save_path)

    # Build feature embedding reconstruction networks (if alpha > 0, beta > 0)
    nets += mh.build_feat_emb_reconst_nets(
            [alpha, beta], n_samples_unsup, n_hidden_u,
            [n_hidden_t_enc, n_hidden_t_dec],
            nets, [encoder_net_init, decoder_net_init])

    # Supervised network
    discrim_net, hidden_rep = mh.build_discrim_net(
        batch_size, n_feats, input_var_sup, n_hidden_t_enc,
        n_hidden_s, embeddings[0], disc_nonlinearity, n_targets)

    # Reconstruct network
    nets += [mh.build_reconst_net(hidden_rep, embeddings[1] if
                                  len(embeddings) > 1
                                  else None, n_feats, gamma)]

    # Load best model
    with np.load(os.path.join(model_path, 'model_feat_sel_best.npz')) as f:
        param_values = [f['arr_%d' % i]
                        for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(filter(None, nets) +
                                        [discrim_net],
                                        param_values)

    print("Building and compiling training functions")

    # Build and compile training functions
    predictions, predictions_det = mh.define_predictions(nets, start=2)
    prediction_sup, prediction_sup_det = mh.define_predictions([discrim_net])
    prediction_sup = prediction_sup[0]
    prediction_sup_det = prediction_sup_det[0]

    # Define losses
    # reconstruction losses
    _, reconst_losses_det = mh.define_reconst_losses(
        predictions, predictions_det, [input_var_unsup, input_var_unsup,
                                       input_var_sup])
    # supervised loss
    _, sup_loss_det = mh.define_sup_loss(
        disc_nonlinearity, prediction_sup, prediction_sup_det, keep_labels,
        target_var_sup, missing_labels_val)

    # Define inputs
    inputs = [input_var_sup, target_var_sup]

    # Combine losses
    loss_det = sup_loss_det + alpha*reconst_losses_det[0] + \
        beta*reconst_losses_det[1] + gamma*reconst_losses_det[2]

    # Define parameters
    params = lasagne.layers.get_all_params(
        [discrim_net] + filter(None, nets), trainable=True)

    l2_penalty = apply_penalty(params, l2)
    loss_det = loss_det + lmd*l2_penalty

    # Monitoring Labels
    monitor_labels = ["reconst. feat. W_enc",
                      "reconst. feat. W_dec",
                      "reconst. loss"]
    monitor_labels = [i for i, j in zip(monitor_labels, reconst_losses_det)
                      if j != 0]
    monitor_labels += ["feat. W_enc. mean", "feat. W_enc var"]
    monitor_labels += ["feat. W_dec. mean", "feat. W_dec var"] if \
        (embeddings[1] is not None) else []
    monitor_labels += ["loss. sup.", "total loss"]

    # Build and compile test function
    val_outputs = reconst_losses_det
    val_outputs = [i for i, j in zip(val_outputs, reconst_losses_det) if j != 0]
    val_outputs += [embeddings[0].mean(), embeddings[0].var()]
    val_outputs += [embeddings[1].mean(), embeddings[1].var()] if \
        (embeddings[1] is not None) else []
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
    test_minibatches = mlh.iterate_minibatches(x_test, y_test, batch_size,
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

    # plt.imshow(cm)
    # plt.show()


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
                        default='/data/lisatmp4/romerosa/datasets/1000_Genome_project/unsupervised_hist_3x26_fold0.npy',
                        # '/data/lisatmp4/romerosa/feature_selection/1000_genomes/kmeans_10_embedding.npy',
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
                        default=0.0001,
                        help="""Float to indicate learning rate.""")
    parser.add_argument('--learning_rate_annealing',
                        '-lra',
                        type=float,
                        default=.99,
                        help="Float to indicate learning rate annealing rate.")
    parser.add_argument('--alpha',
                        '-a',
                        type=float,
                        default=0.,
                        help="""reconst_loss coeff. for auxiliary net W_enc""")
    parser.add_argument('--beta',
                        '-b',
                        type=float,
                        default=0.,
                        help="""reconst_loss coeff. for auxiliary net W_dec""")
    parser.add_argument('--gamma',
                        '-g',
                        type=float,
                        default=10.,
                        help="""reconst_loss coeff. (used for aux net W-dec as well)""")
    parser.add_argument('--lmd',
                        '-l',
                        type=float,
                        default=.0,
                        help="""Weight decay coeff.""")
    parser.add_argument('--disc_nonlinearity',
                        '-nl',
                        default="softmax",
                        help="""Nonlinearity to use in disc_net's last layer""")
    parser.add_argument('--encoder_net_init',
                        '-eni',
                        type=float,
                        default=0.01,
                        help="Bounds of uniform initialization for " +
                             "encoder_net weights")
    parser.add_argument('--decoder_net_init',
                        '-dni',
                        type=float,
                        default=0.01,
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
    parser.add_argument('-embedding_input',
                        type=str,
                        default='raw',
                        help='The kind of input we will use for the feat. emb. nets')
    parser.add_argument('--model_path',
                        default='/data/lisatmp4/erraqaba/feature_selection/',
                        help='Path to save results.')
    parser.add_argument('--save_path',
                        default='/data/lisatmp4/'+ os.environ["USER"]+'/feature_selection/',
                        help='Path to save results.')
    parser.add_argument('--dataset_path',
                        default='/data/lisatmp4/romerosa/datasets/1000_Genome_project/',
                        help='Path to dataset')
    parser.add_argument('-resume',
                        type=bool,
                        default=False,
                        help='Whether to resume job')
    parser.add_argument('-exp_name',
                        type=str,
                        default='final_unsupervisedhist3x26fold0__new_our_model1.0_raw_lr-0.0001_anneal-0.99_eni-0.01_dni-0.01_accuracy_Ri10.0_hu-100_tenc-100_tdec-100_hs-100_fold0',
                        help='Experiment name that will be concatenated at the beginning of the generated name')

    args = parser.parse_args()
    print ("Printing args")
    print (args)

    execute(args.dataset,
            mlh.parse_int_list_arg(args.n_hidden_u),
            mlh.parse_int_list_arg(args.n_hidden_t_enc),
            mlh.parse_int_list_arg(args.n_hidden_t_dec),
            mlh.parse_int_list_arg(args.n_hidden_s),
            args.embedding_source,
            int(args.num_epochs),
            args.learning_rate,
            args.learning_rate_annealing,
            args.alpha,
            args.beta,
            args.gamma,
            args.lmd,
            args.disc_nonlinearity,
            args.encoder_net_init,
            args.decoder_net_init,
            args.keep_labels,
            args.prec_recall_cutoff != 0, -1,
            args.which_fold,
            args.early_stop_criterion,
            args.embedding_input,
            args.model_path,
            args.save_path,
            args.dataset_path,
            args.resume,
            args.exp_name)


if __name__ == '__main__':
    main()
