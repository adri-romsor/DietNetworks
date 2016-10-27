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
from lasagne.regularization import apply_penalty, l2
import numpy as np
import theano
from theano import config
import theano.tensor as T

# import mainloop_helpers as mlh
# import model_helpers as mh

import random

import ipdb
print ("config floatX: {}".format(config.floatX))


# creating data generator
def data_generator(dataset, batch_size):
    while True:
        x_list, y_list, mask_index = [], [], []
        for i in range(batch_size):
            index_feat = random.randint(0, dataset.shape[1]/2-1)
            index_individual = random.randint(0, dataset.shape[0]/2-1)
            datamod = dataset[index_individual, :]
            # ipdb.set_trace()
            datamod = np.concatenate([
                    datamod[:index_feat],
                    np.array([0, 0]),
                    datamod[index_feat+2:]])

            target = tuple(dataset[index_individual, index_feat:index_feat+2])
            mask_index += np.array([index_feat, index_feat+1])

            x_list.append(datamod)
            y_list.append(target)

        yield(np.stack(x_list), np.stack(y_list), np.stack(mask_index))

def main(learning_rate=0.00001, alpha=.5, save_path=None, save_copy=None):

    # Reading dataset
    base_data_path = "/data/lisatmp4/romerosa/datasets/1000_Genome_project/"
    dataset_path = os.path.join(
            base_data_path,
            "unsupervised_snp_bin_fold0.npy")

    dataset = np.load(dataset_path)
    n_features = dataset.shape[1]

    save_copy = '/Tmp/'+ os.environ["USER"]+'/feature_selection/'
    save_path = '/data/lisatmp4/'+ os.environ["USER"]+'/feature_selection/'

    exp_name = "learn_gene_vector"
    save_path = os.path.join(save_path, exp_name)
    save_copy = os.path.join(save_copy, exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_copy):
        os.makedirs(save_copy)

    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('input')
    target_var = T.matrix('target')
    target_reconst = T.matrix('target')
    lr = theano.shared(np.float32(learning_rate), 'learning_rate')
    lmd = 0.0001  # weight decay coeff
    num_epochs = 200
    # there arent really any epochs as we are using a generator with random
    # sampling from dataset. This is for compat.
    batches_per_epoch = 1000
    batch_size = 128

    # building network
    encoder = InputLayer((batch_size, n_features), input_var)
    # for i, out in enumerate(n_hidden_u):

    # building the encoder and decoder
    reduction_fraction = [.02, .01, .001]
    for i in range(len(reduction_fraction)):
        encoder = DenseLayer(
                encoder,
                num_units=reduction_fraction[i]*n_features,
                nonlinearity=rectify)
    for i in range(len(reduction_fraction)):
        if i == 0:
            decoder = DenseLayer(
                    encoder,
                    num_units=reduction_fraction[::-1][i]*n_features,
                    nonlinearity=rectify)
        else:
            decoder = DenseLayer(
                    decoder,
                    num_units=reduction_fraction[::-1][i]*n_features,
                    nonlinearity=rectify)
        decoder = DenseLayer(
                decoder,
                num_units=n_features,
                nonlinearity=sigmoid)

    predictor_laysize = (reduction_fraction[-1]*n_features)*4
    for i in range(len(predictor_laysize)):
        predictor = DenseLayer(
                encoder if i == 0 else predictor,
                num_units=predictor_laysize[i],
                nonlinearity=rectify)

    predictor = DenseLayer(
            predictor,
            num_units=2,
            nonlinearity=sigmoid)

    prediction_reconst = lasagne.layers.get_output(decoder)
    prediction_var = lasagne.layers.get_output(predictor)
    loss1 = lasagne.objectives.binary_crossentropy(
        prediction_reconst, target_reconst)

    loss2 = lasagne.objectives.binary_crossentropy(
        prediction_var, target_var
    )

    params = lasagne.layers.get_all_params(
        [predictor, decoder], trainable=True)

    # Combine losses
    loss = loss1 + alpha * loss2

    # applying weight decay
    l2_penalty = apply_penalty(params, l2)
    loss = loss + lmd*l2_penalty
    loss = loss + lmd*l2_penalty

    # Some variables
    # max_patience = 100
    # patience = 0

    train_monitored = []
    valid_monitored = []
    train_loss = []

    updates = lasagne.updates.rmsprop(loss,
                                      params,
                                      learning_rate=lr)

    inputs = [input_var, target_var, target_reconst]
    # Compile training function
    print "Compiling training function"
    train_fn = theano.function(inputs, loss, updates=updates,
                               on_unused_input='ignore')

    start_training = time.time()
    print "training start time: {}".format(start_training)

    # Creating dataset
    print "Creating dataset"
    data_generator(dataset, batch_size)

    print "Starting training"
    for epoch in range(num_epochs):
        start_time = time.time()
        print("Epoch {} of {}".format(epoch+1, num_epochs))
        nb_minibatches = 0
        loss_epoch = 0

        # Train pass
        for batch_index in range(batches_per_epoch):
            batch = data_generator.next()
            ipdb.set_trace()
            loss_epoch += train_fn(*batch)
            nb_minibatches += 1

        loss_epoch /= nb_minibatches
        train_loss += [loss_epoch]

        if epoch % 5 == 0:
            # Save stuff
            np.savez(os.path.join(save_path, 'model_gene_vector_last.npz'),
                     *lasagne.layers.get_all_param_values(filter(None, nets) +
                                                          [discrim_net]))
            np.savez(save_path + "errors_gene_vector.npz",
                     zip(*train_monitored), zip(*valid_monitored))

    # Print all final errors for train, validation and test
    print("Training time:\t\t\t{:.3f}s".format(time.time() - start_training))

    # Copy files to loadpath
    if save_path != save_copy:
        print('Copying model and other training files to {}'.format(save_copy))
        copy_tree(save_path, save_copy)


if __name__ == '__main__':
    main()
