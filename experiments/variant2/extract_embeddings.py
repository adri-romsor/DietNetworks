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

from DietNetworks.experiments.common import dataset_utils

import matplotlib.pyplot as plt

import mainloop_helpers as mlh
import model_helpers as mh


# Main program
def execute(dataset, n_hidden_u, n_hidden_t_enc, n_hidden_t_dec, n_hidden_s,
            learning_rate, learning_rate_annealing=.99, embedding_source=None,
            alpha=1, beta=1, gamma=1, lmd=0.0, encoder_net_init=0.1,
            decoder_net_init=0.1, keep_labels=1.0, which_fold=0,
            early_stop_criterion='accuracy', exp_name='', representation='features',
            which_set='test',
            model_path='/Tmp/romerosa/DietNetworks/newmodel/',
            save_path='/Tmp/romerosa/DietNetworks/',
            dataset_path='/Tmp/' + os.environ["USER"] + '/datasets/'):

    print(save_path)

    # Prepare embedding information
    if embedding_source is None:
        embedding_input = 'raw'
    else:
        embedding_input = embedding_source
        embedding_source = os.path.join(dataset_path, embedding_input + '_fold' + str(which_fold) + '.npy')

    # Load the dataset
    print("Loading data")
    x_train, y_train, x_valid, y_valid, x_test, y_test, \
        x_unsup, training_labels = mlh.load_data(
            dataset, dataset_path, embedding_source,
            which_fold=which_fold, keep_labels=1.0,
            missing_labels_val=-1.0,
            embedding_input=embedding_input)

    if which_set == 'train':
        x = x_train
        y = y_train
    elif which_set == 'valid':
        x = x_valid
        y = y_valid
    elif which_set == 'test':
        x = x_test
        y = y_test

    if x_unsup is not None:
        n_samples_unsup = x_unsup.shape[1]
    else:
        n_samples_unsup = 0

    # Extract required information from data
    n_samples, n_feats = x_train.shape
    n_targets = y_train.shape[1]

    # Set some variables
    batch_size = 1
    beta = gamma if (gamma == 0) else beta

    # Preparing folder to save stuff
    if embedding_source is None:
        embedding_name = embedding_input
    else:
        embedding_name = embedding_source.replace("_", "").split(".")[0]
        exp_name += embedding_name.rsplit('/', 1)[::-1][0] + '_'

    exp_name += mlh.define_exp_name(keep_labels, alpha, beta, gamma, lmd,
                                    n_hidden_u, n_hidden_t_enc, n_hidden_t_dec,
                                    n_hidden_s, which_fold,
                                    learning_rate, decoder_net_init,
                                    encoder_net_init, early_stop_criterion,
                                    learning_rate_annealing)

    print("Experiment: " + exp_name)
    model_path = os.path.join(model_path, dataset, exp_name)
    print(model_path)
    save_path = os.path.join(save_path, representation, embedding_input,
                             'fold' + str(which_fold))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Prepare Theano variables for inputs and targets
    input_var_sup = T.matrix('input_sup')
    input_var_unsup = theano.shared(x_unsup, 'input_unsup')  # x_unsup TBD
    target_var_sup = T.matrix('target_sup')

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
        encoder_net_init, save_path)

    # Build feature embedding reconstruction networks (if alpha > 0, beta > 0)
    nets += mh.build_feat_emb_reconst_nets(
            [alpha, beta], n_samples_unsup, n_hidden_u,
            [n_hidden_t_enc, n_hidden_t_dec],
            nets, [encoder_net_init, encoder_net_init])

    # Supervised network
    discrim_net, hidden_rep = mh.build_discrim_net(
        batch_size, n_feats, input_var_sup, n_hidden_t_enc,
        n_hidden_s, embeddings[0], 'softmax', n_targets)

    # Reconstruct network
    nets += [mh.build_reconst_net(hidden_rep, embeddings[1] if
                                  len(embeddings) > 1
                                  else None, n_feats, gamma)]

    # Load best model
    with np.load(os.path.join(model_path, 'dietnets_best.npz')) as f:
        param_values = [f['arr_%d' % i]
                        for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(filter(None, nets) +
                                        [discrim_net],
                                        param_values)

    print("Building and compiling training functions")

    # Build and compile training functions
    if representation == 'features':
        feat_layers = lasagne.layers.get_all_layers(nets[0])
        predictions = lasagne.layers.get_output(feat_layers)
        inputs = []
        predict = theano.function(inputs, predictions)
        all_pred = predict()
        all_pred = all_pred

        for i, el in enumerate(all_pred):
            file_name = os.path.join(save_path, 'layer'+str(i)+'.npy')
            print(file_name)
            np.save(file_name, el)

    elif representation == 'subjects':
        subject_layers = lasagne.layers.get_all_layers(discrim_net)
        subject_layers = [el for el in subject_layers if isinstance(el, DenseLayer)]
        predictions = lasagne.layers.get_output(subject_layers)
        inputs = [input_var_sup]
        predict = theano.function(inputs, predictions)

        iterate_minibatches = mlh.iterate_minibatches(x, y, batch_size,
                                                      shuffle=False)
        print("Starting testing...")
        all_pred = []
        for batch in iterate_minibatches:
            all_pred += [predict(batch[0])]

        all_pred = zip(*all_pred)
        all_pred = [np.vstack(el) for el in all_pred]

        for i, el in enumerate(all_pred):
            file_name = os.path.join(save_path, 'layer'+str(i)+'_'+which_set+'.npz')
            print(file_name)
            np.savez(file_name,
                     representation=el,
                     label=y.argmax(1))


def main():
    parser = argparse.ArgumentParser(description="""Extract embeddings""")
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
    parser.add_argument('--embedding_source',
                        default='histo3x26',
                        help='Source for the feature embedding. Either' +
                             'None or the name of a file from which' +
                             'to load a learned embedding')
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
    parser.add_argument('--which_fold',
                        type=int,
                        default=0,
                        help='Which fold to use for cross-validation (0-4)')
    parser.add_argument('--early_stop_criterion',
                        default='accuracy',
                        help='What monitored variable to use for early-stopping')
    parser.add_argument('-exp_name',
                        type=str,
                        default='dietnets_final_',
                        help='Experiment name that will be concatenated at the beginning of the generated name')
    parser.add_argument('-representation',
                        type=str,
                        default='subjects',
                        help='features or subjects')
    parser.add_argument('-which_set',
                        type=str,
                        default='test',
                        help='features or subjects')
    parser.add_argument('--model_path',
                        default='/data/lisatmp4/'+os.environ["USER"]+'/DietNetworks/',
                        help='Path to save results.')
    parser.add_argument('--save_path',
                        default='/data/lisatmp4/'+ os.environ["USER"]+'/DietNetworks/',
                        help='Path to save results.')
    parser.add_argument('--dataset_path',
                        default='/data/lisatmp4/romerosa/datasets/1000_Genome_project/',
                        help='Path to dataset')

    args = parser.parse_args()
    print ("Printing args")
    print (args)

    folds = [0, 1, 2, 3, 4]

    for f in folds:
        execute(args.dataset,
                mlh.parse_int_list_arg(args.n_hidden_u),
                mlh.parse_int_list_arg(args.n_hidden_t_enc),
                mlh.parse_int_list_arg(args.n_hidden_t_dec),
                mlh.parse_int_list_arg(args.n_hidden_s),
                args.learning_rate,
                args.learning_rate_annealing,
                args.embedding_source,
                args.alpha,
                args.beta,
                args.gamma,
                args.lmd,
                args.encoder_net_init,
                args.decoder_net_init,
                args.keep_labels,
                f,
                args.early_stop_criterion,
                args.exp_name,
                args.representation,
                'train',
                args.model_path,
                args.save_path,
                args.dataset_path)


if __name__ == '__main__':
    main()
