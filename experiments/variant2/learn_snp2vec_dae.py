#!/usr/bin/env python
# from __future__ import print_function
import argparse
import time
import os
from distutils.dir_util import copy_tree

import lasagne
from lasagne.layers import DenseLayer, InputLayer
from lasagne.nonlinearities import (sigmoid, softmax, tanh, linear, rectify,
                                    leaky_rectify, very_leaky_rectify)
from lasagne.regularization import apply_penalty, l2, l1
from lasagne.init import (Uniform, GlorotUniform, GlorotNormal, Normal, He,
                          HeNormal, HeUniform)
import numpy as np
import theano
from theano import config
import theano.tensor as T

import mainloop_helpers as mlh
import model_helpers as mh
from mainloop_helpers import parse_string_int_tuple

import random
import getpass

import json

# import ipdb
print ("config floatX: {}".format(config.floatX))

username = getpass.getuser()
if username == "sylvaint" and not os.path.isdir("/Tmp/sylvaint"):
    os.makedirs("/Tmp/sylvaint")


# creating data generator
def data_generator(dataset, batch_size, shuffle=False, noise=0.5):
    nb_subjects = dataset.shape[0]
    nb_feats = dataset.shape[1]

    if shuffle:
        indices = np.random.permutation(nb_subjects)
    else:
        indices = np.arange(nb_subjects)

    for i in range(0, nb_subjects - batch_size + 1, batch_size):
        inputs = dataset[indices[i:i + batch_size], :]
        n_in_batch = inputs.shape[0]

        # Keep a copy of the original inputs to act as a reconstruction target
        reconstruction_targets = inputs.copy()

        nb_SNPs = nb_feats / 2
        snp_mask = np.random.binomial(1, 1-noise,
                                      (n_in_batch, nb_SNPs)).astype("uint8")
        inputs[:, ::2] *= snp_mask
        inputs[:, 1::2] *= snp_mask

        yield inputs, reconstruction_targets


def convert_initialization(component, nonlinearity="sigmoid"):
    # component = init_dic[component_key]
    assert(len(component) == 2)
    if component[0] == "uniform":
        return Uniform(component[1])
    elif component[0] == "glorotnormal":
        if nonlinearity in ["linear", "sigmoid", "tanh"]:
            return GlorotNormal(1.)
        else:
            return GlorotNormal("relu")
    elif component[0] == "glorotuniform":
        if nonlinearity in ["linear", "sigmoid", "tanh"]:
            return GlorotUniform(1.)
        else:
            return GlorotUniform("relu")
    elif component[0] == "normal":
        return Normal(*component[1])
    else:
        raise NotImplementedError()


def execute(dataset, learning_rate=0.00001, learning_rate_annealing=1.0,
            lmd=0., noise=0.0, encoder_units=[1024, 512, 256],
            num_epochs=500, which_fold=1,
            save_path=None, save_copy=None, dataset_path=None,
            num_fully_connected=0, exp_name='', init_args=None):

    # Reading dataset
    print("Loading data")
    if dataset == "1000_genomes" and which_fold == 1 and False:
        x_unsup = mlh.load_data(dataset, dataset_path, None,
                                which_fold=which_fold, keep_labels=1.0,
                                missing_labels_val=-1.0,
                                embedding_input='raw', transpose=False)
        import pdb; pdb.set_trace()

        x_train = np.zeros((x_unsup[0].shape[0], x_unsup[0].shape[1]*2), dtype="int8")
        x_train[:,::2] = (x_unsup[0] == 2)
        x_train[:,1::2] = (x_unsup[0] >= 1)

        x_valid = np.zeros((x_unsup[2].shape[0], x_unsup[2].shape[1]*2), dtype="int8")
        x_valid[:,::2] = (x_unsup[2] == 2)
        x_valid[:,1::2] = (x_unsup[2] >= 1)
    else:
        x_unsup = mlh.load_data(dataset, dataset_path, None,
                                which_fold=which_fold, keep_labels=1.0,
                                missing_labels_val=-1.0,
                                embedding_input='bin', transpose=True)
        x_train = x_unsup[0][0]
        x_valid = x_unsup[1][0]

    print(x_train.shape, x_valid.shape)

    n_features = x_train.shape[1]

    exp_name += "learn_snp2vec_dae_h"
    for e in encoder_units:
        exp_name += ('-' + str(e))
    # exp_name += '_g-' + str(gamma)
    exp_name += '_l-' + str(lmd)
    exp_name += '_lr-' + str(learning_rate)
    exp_name += '_fold-' + str(which_fold)

    save_path = os.path.join(save_path, exp_name)
    save_copy = os.path.join(save_copy, exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_copy):
        os.makedirs(save_copy)

    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('input')
    target_reconst = T.matrix('target')
    lr = theano.shared(np.float32(learning_rate), 'learning_rate')
    batch_size = 128

    # building network
    encoder = InputLayer((batch_size, n_features), input_var)

    # building the encoder and decoder
    #import pdb; pdb.set_trace()
    for i in range(len(encoder_units)):
        encoder = DenseLayer(
                encoder,
                num_units=encoder_units[i],
                W=Uniform(0.00001),
                nonlinearity=leaky_rectify)  # if i < len(encoder_units)-1 else linear)

    embedding = lasagne.layers.get_output(encoder)
    get_embedding_fn = theano.function([input_var], embedding)

    params = lasagne.layers.get_all_params(encoder, trainable=True)
    monitor_labels = ["embedding min", "embedding mean", "embedding max"]
    val_outputs = [embedding.min(), embedding.mean(), embedding.max()]
    nets = [encoder]

    decoder_units = encoder_units[::-1][1:]
    print(decoder_units)
    decoder = encoder
    for i in range(len(decoder_units)):
        decoder = DenseLayer(decoder,
                             num_units=decoder_units[i],
                             W=Uniform(0.0001),
                             nonlinearity=leaky_rectify)
    decoder = DenseLayer(decoder,
                         num_units=n_features,
                         W=convert_initialization(
                                init_args["decoder_init"],
                                nonlinearity="sigmoid"),
                         nonlinearity=sigmoid)
    prediction_reconst = lasagne.layers.get_output(decoder)

    # Reconstruction error
    loss_reconst = lasagne.objectives.binary_crossentropy(prediction_reconst,
                                                          target_reconst).mean()

    # loss_reconst = mh.define_sampled_mean_bincrossentropy(
    #    prediction_reconst, target_reconst, gamma=gamma)

    #loss_reconst = mh.dice_coef_loss(
    #    target_reconst, prediction_reconst).mean()

    accuracy = T.eq(T.gt(prediction_reconst, 0.5), target_reconst).mean()

    params += lasagne.layers.get_all_params(decoder, trainable=True)
    monitor_labels += ["reconst. loss", "reconst. accuracy"]
    val_outputs += [loss_reconst, accuracy]
    nets += [decoder]
    # sparsity_reconst = gamma * l1(prediction_reconst)
    # roh = input_var.mean(0)
    # sparsity_reconst = ((roh * T.log(roh / (prediction_reconst.mean(0)+1e-8))) +\
    #     ((1 - roh) * T.log((1 - roh) / (1 - prediction_reconst + 1e-8)))).sum()

    # Combine losses
    loss = loss_reconst # + sparsity_reconst

    # applying weight decay
    l2_penalty = apply_penalty(params, l2)
    loss = loss + lmd * l2_penalty

    val_outputs += [loss]
    monitor_labels += ['loss']

    # Some variables
    max_patience = 100
    patience = 0

    train_monitored = []
    valid_monitored = []
    train_loss = []

    updates = lasagne.updates.adam(loss,
                                   params,
                                   learning_rate=lr)

    for k in updates.keys():
        if updates[k].ndim == 2:
            updates[k] = lasagne.updates.norm_constraint(updates[k], 1.0)

    inputs = [input_var, target_reconst]

    # Compile training function
    print "Compiling training function"
    train_fn = theano.function(inputs, loss, updates=updates,
                               on_unused_input='ignore')
    val_fn = theano.function(inputs,
                             [val_outputs[0]] + val_outputs,
                             on_unused_input='ignore')

    start_training = time.time()

    print "Starting training"
    for epoch in range(num_epochs):
        start_time = time.time()
        print("Epoch {} of {}".format(epoch+1, num_epochs))
        nb_minibatches = 0
        loss_epoch = 0

        for x, target_reconst_val in data_generator(x_train, batch_size,
                                                    shuffle=True, noise=noise):
            loss_epoch += train_fn(x, target_reconst_val)
            nb_minibatches += 1

        loss_epoch /= nb_minibatches
        train_loss += [loss_epoch]

        # Monitoring on the training set
        train_minibatches = data_generator(x_train, batch_size, noise=noise)
        train_err = mlh.monitoring(train_minibatches, "train", val_fn,
                                   monitor_labels, 0)
        train_monitored += [train_err]

        # Monitoring on the validation set
        valid_minibatches = data_generator(x_valid, batch_size, noise=noise)

        valid_err = mlh.monitoring(valid_minibatches, "valid", val_fn,
                                   monitor_labels, 0)
        valid_monitored += [valid_err]

        early_stop_criterion = 'loss'
        early_stop_val = valid_err[monitor_labels.index(early_stop_criterion)]

        # Early stopping
        if epoch == 0:
            best_valid = early_stop_val
        elif early_stop_val < best_valid and early_stop_criterion == 'loss':
            best_valid = early_stop_val
            patience = 0

            # Save stuff
            np.savez(save_path+'/model_snp2vec_best.npz',
                     *lasagne.layers.get_all_param_values(nets))
            np.savez(save_path + "/errors_snp2vec_best.npz",
                     zip(*train_monitored), zip(*valid_monitored))
        else:
            patience += 1
            np.savez(os.path.join(save_path, 'model_snp2vec_last.npz'),
                     *lasagne.layers.get_all_param_values(nets))
            np.savez(save_path + "/errors_snp2vec_last.npz",
                     zip(*train_monitored), zip(*valid_monitored))

        # End training
        if (patience == max_patience) or (epoch == num_epochs-1):
            print("Ending training")
            # Load best model
            if not os.path.exists(save_path + '/model_snp2vec_best.npz'):
                print("No saved model to be tested and/or generate"
                      " the embedding !")
            else:
                with np.load(save_path + '/model_snp2vec_best.npz') as f:
                    param_values = [f['arr_%d' % i]
                                    for i in range(len(f.files))]
                    lasagne.layers.set_all_param_values(nets, param_values)

            # Use the saved model to generate the feature embedding
            # Here the feature embedding is the different in the hidden
            # representation between having that feature on and having it off
            print("Generating embedding")
            embedding_size = encoder_units[-1]
            null_input = np.zeros((1, n_features), dtype="float32")
            null_embedding = get_embedding_fn(null_input)[0]

            all_embeddings = np.zeros((n_features,
                                       embedding_size), dtype="float32")

            """
            single_feat_input = null_input.copy()
            for i in range(n_features):
                if i % 10000 == 0:
                    print(i, n_features)

                single_feat_input[:,i] = 1
                all_embeddings[i] = (get_embedding_fn(single_feat_input)[0] -
                                     null_embedding)
                single_feat_input[:,i] = 0

            result1 = all_embeddings[:1000].copy()
            """

            block_size = 10
            single_feat_batch = np.zeros((block_size, n_features), dtype="float32")
            for i in range(0, n_features, block_size):
                if i % 10000 == 0:
                    print(i, n_features)

                for j in range(block_size):
                    single_feat_batch[j, i+j] = 1

                all_embeddings[i:i+10] = (get_embedding_fn(single_feat_batch) -
                                          null_embedding)

                for j in range(block_size):
                    single_feat_batch[j, i+j] = 0

            np.save("/Tmp/carriepl/DietNetworks/all_embeddings_noise%f_fold%i.npy" % (which_fold, noise),
                    all_embeddings)

            # Training set results
            train_minibatches = data_generator(x_train, batch_size, noise=noise)
            train_err = mlh.monitoring(train_minibatches, "train", val_fn,
                                       monitor_labels, 0)

            # Validation set results
            valid_minibatches = data_generator(x_valid, batch_size, noise=noise)
            valid_err = mlh.monitoring(valid_minibatches, "valid", val_fn,
                                       monitor_labels, 0)

            # Stop
            print("  epoch time:\t\t\t{:.3f}s \n".format(time.time() -
                                                         start_time))
            break

        print("  epoch time:\t\t\t{:.3f}s \n".format(time.time() - start_time))
        # Anneal the learning rate
        lr.set_value(float(lr.get_value() * learning_rate_annealing))


    # Copy files to loadpath
    if save_path != save_copy:
        print('Copying model and other training files to {}'.format(save_copy))
        copy_tree(save_path, save_copy)


def main():
    parser = argparse.ArgumentParser(description="""Train snp2vec embedding.""")
    parser.add_argument('--dataset',
                        default='1000_genomes',
                        help='Dataset.')
    parser.add_argument('--learning_rate',
                        '-lr',
                        type=float,
                        default=.001,
                        help="""Float to indicate learning rate.""")
    parser.add_argument('--learning_rate_annealing',
                        '-lra',
                        type=float,
                        default=.99,
                        help="Float to indicate learning rate annealing rate.")
    parser.add_argument('--lmd',
                        '-l',
                        type=float,
                        default=.0,
                        help="""Float to indicate weight decay coeff.""")
    parser.add_argument('--noise',
                        type=float,
                        default=.25,
                        help="Float to indicate fraction of inputs to blackout.")
    # parser.add_argument('--gamma',
    #                     '-g',
    #                     type=float,
    #                     default=.0005,
    #                     help="""Float to indicate l1 penalty.""")
    parser.add_argument('--encoder_init',
                        '-eni',
                        default=("uniform",0.01),
                        help="Initialization for " +
                             "encoder_net weights")
    parser.add_argument('--decoder_init',
                        '-dei',
                        default=['glorotnormal', 1.0],
                        help="Initialization for " +
                             "decoder_net weights")
    parser.add_argument('--predictor_init',
                        '-pri',
                        default=['glorotnormal', 1.0],
                        help="Initialization for " +
                             "predictor_net weights")
    parser.add_argument('--encoder_units',
                        default=[100, 100],
                        help='List of encoder hidden units.')
    parser.add_argument('--num_epochs',
                        '-ne',
                        type=int,
                        default=1000,
                        help="""Int to indicate the max'
                        'number of epochs.""")
    parser.add_argument('--which_fold',
                        type=int,
                        default=0,
                        help='Which fold to use for cross-validation (0-4)')
    parser.add_argument('--save_tmp',
                        default='/Tmp/'+os.environ["USER"] +
                                '/DietNetworks/',
                        help='Path to save results.')
    parser.add_argument('--save_perm',
                        default='/data/lisatmp4/'+os.environ["USER"] +
                                '/DietNetworks/',
                        help='Path to save results.')
    parser.add_argument('--dataset_path',
                        default='/data/lisatmp4/romerosa/datasets/1000_Genome_project/',
                        help='Path to dataset')
    parser.add_argument('--num_fully_connected',
                        type=int,
                        default=0,
                        help='Number of fully connected layers in predictor')
    parser.add_argument('-exp_name',
                        type=str,
                        default='shuffle_',
                        help='Experiment name that will be concatenated at\
                            the beginning of the generated name')

    args = parser.parse_args()

    # import ipdb; ipdb.set_trace()

    init_args = {"encoder_init" : parse_string_int_tuple(args.encoder_init),
                 "decoder_init" : parse_string_int_tuple(args.decoder_init),
                 "predictor_init" : parse_string_int_tuple(args.predictor_init)}

    print args
    print "init_args: {}".format(init_args)

    execute(args.dataset,
            args.learning_rate,
            args.learning_rate_annealing,
            args.lmd,
            args.noise,
            # args.gamma,
            # args.encoder_net_init,
            mlh.parse_int_list_arg(args.encoder_units),
            int(args.num_epochs),
            int(args.which_fold),
            args.save_tmp,
            args.save_perm,
            args.dataset_path,
            args.num_fully_connected,
            args.exp_name,
            init_args)


if __name__ == '__main__':
    main()
