#!/usr/bin/env python2

from __future__ import print_function
import argparse
import time
import os
import tables
from distutils.dir_util import copy_tree

import lasagne
from lasagne.layers import DenseLayer, InputLayer, DropoutLayer, ReshapeLayer, \
    NonlinearityLayer
from lasagne.nonlinearities import sigmoid, softmax, tanh, linear, rectify
from lasagne.regularization import apply_penalty, l2, l1
from lasagne.init import Uniform
import numpy as np
import theano
import theano.tensor as T

from DietNetworks.experiments.common import dataset_utils

import mainloop_helpers as mlh
import model_helpers as mh


# Main program
def execute(dataset, n_hidden_u, num_epochs=500,
            learning_rate=.001, learning_rate_annealing=1.0, lmd=.0001,
            embedding_input='raw', which_fold=0,
            save_path='/Tmp/$USER/DietNetworks/newmodel/',
            save_copy='/Tmp/$USER/DietNetworks/newmodel/',
            dataset_path='/Tmp/$USER/DietNetworks/newmodel/'):

    # Load the dataset
    print("Loading data")
    x_unsup = mlh.load_data(
            dataset, dataset_path, None,
            which_fold=which_fold, keep_labels=1.0,
            missing_labels_val=-1.0,
            embedding_input=embedding_input, transpose=True)

    x_train = x_unsup[0][0]
    x_valid = x_unsup[1][0]

    # Extract required information from data
    n_row, n_col = x_train.shape
    print('Data size ' + str(n_row) + 'x' + str(n_col))

    # Set some variables
    batch_size = 256

    # Define experiment name
    exp_name = 'pretrain_' + mlh.define_exp_name(1., 0, 0, 0, lmd,
                                                 n_hidden_u, [], [], [],
                                                 which_fold, embedding_input,
                                                 learning_rate, 0, 0,
                                                 'reconst_loss',
                                                 learning_rate_annealing)
    print('Experiment: ' + exp_name)

    # Preparing folder to save stuff
    save_path = os.path.join(save_path, dataset, exp_name)
    save_copy = os.path.join(save_copy, dataset, exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('input_unsup')
    lr = theano.shared(np.float32(learning_rate), 'learning_rate')

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

    decoder_net = encoder_net
    for i in range(len(n_hidden_u)-2, -1, -1):
        decoder_net = DenseLayer(decoder_net, num_units=n_hidden_u[i],
                                 nonlinearity=linear)
        decoder_net = DropoutLayer(decoder_net)

    decoder_net = DenseLayer(decoder_net, num_units=n_col,
                             nonlinearity=linear)

    if embedding_input == 'raw' or embedding_input == 'w2v':
        final_nonlin = linear
    elif embedding_input == 'bin':
        final_nonlin = sigmoid
    elif 'histo' in embedding_input:
        final_nonlin = softmax

    if embedding_input == 'histo3x26':
        laySize = lasagne.layers.get_output(decoder_net).shape
        decoder_net = ReshapeLayer(decoder_net, (laySize[0]*26, 3))

    decoder_net = NonlinearityLayer(decoder_net, nonlinearity=final_nonlin)

    if embedding_input == 'histo3x26':
        decoder_net = ReshapeLayer(decoder_net, (laySize[0], laySize[1]))

    print("Building and compiling training functions")
    # Build and compile training functions
    predictions, predictions_det = mh.define_predictions(
        [encoder_net, decoder_net], start=0)
    prediction_sup, prediction_sup_det = mh.define_predictions(
        [encoder_net, decoder_net], start=0)

    # Define losses
    # reconstruction losses
    loss, loss_det = mh.define_loss(predictions[1], predictions_det[1],
                                    input_var, embedding_input)

    # Define parameters
    params = lasagne.layers.get_all_params(decoder_net, trainable=True)

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

    # Apply norm constraints on the weights
    for k in updates.keys():
        if updates[k].ndim == 2:
            updates[k] = lasagne.updates.norm_constraint(updates[k], 1.0)

    # Compile training function
    train_fn = theano.function([input_var], loss,
                               updates=updates,
                               on_unused_input='ignore')

    # Expressions required for test
    monitor_labels = ['loss']
    val_outputs = [loss_det]

    # Add some monitoring on the learned feature embedding
    val_outputs += [predictions[0].min(), predictions[0].mean(),
                    predictions[0].max(), predictions[0].var()]
    monitor_labels += ["feat. emb. min", "feat. emb. mean",
                       "feat. emb. max", "feat. emb. var"]

    # Compile validation function
    val_fn = theano.function([input_var],
                             val_outputs)

    pred_feat_emb = theano.function([input_var], predictions_det[0])


    # Finally, launch the training loop.
    print("Starting training...")

    # Some variables
    max_patience = 100
    patience = 0

    train_monitored = []
    valid_monitored = []
    train_loss = []

    nb_minibatches = n_row/batch_size
    print("Nb of minibatches: " + str(nb_minibatches))
    start_training = time.time()
    for epoch in range(num_epochs):
        start_time = time.time()
        print("Epoch {} of {}".format(epoch+1, num_epochs))

        loss_epoch = 0

        # Train pass
        for batch in mlh.iterate_minibatches_unsup(x_train, batch_size,
                                                   shuffle=True):
            loss_epoch += train_fn(batch)

        loss_epoch /= nb_minibatches
        train_loss += [loss_epoch]

        train_minibatches = mlh.iterate_minibatches_unsup(x_train, batch_size,
                                                          shuffle=True)
        train_err = mlh.monitoring(train_minibatches, "train", val_fn,
                                   monitor_labels, start=0)
        train_monitored += [train_err]

        # Validation pass
        valid_minibatches = mlh.iterate_minibatches_unsup(x_valid, batch_size,
                                                          shuffle=True)

        valid_err = mlh.monitoring(valid_minibatches, "valid", val_fn,
                                   monitor_labels, start=0)

        valid_monitored += [valid_err]

        try:
            early_stop_val = valid_err[
                monitor_labels.index('loss')]
        except:
            raise ValueError("There is no monitored value by the name of %s" %
                             early_stop_criterion)

        # Eearly stopping
        if epoch == 0:
            best_valid = early_stop_val
        elif early_stop_val < best_valid:
            best_valid = early_stop_val
            patience = 0

            # Save stuff
            np.savez(os.path.join(save_path, 'model_enc_unsupervised_best.npz'),
                     *lasagne.layers.get_all_param_values(encoder_net))
            np.savez(os.path.join(save_path, 'model_ae_unsupervised_best.npz'),
                     *lasagne.layers.get_all_param_values(encoder_net))
            np.savez(os.path.join(save_path, "errors_unsupervised_best.npz"),
                     zip(*train_monitored), zip(*valid_monitored))
        else:
            patience += 1
            # Save stuff
            np.savez(os.path.join(save_path, 'model_enc_unsupervised_last.npz'),
                     *lasagne.layers.get_all_param_values(encoder_net))
            np.savez(os.path.join(save_path, 'model_ae_unsupervised_last.npz'),
                     *lasagne.layers.get_all_param_values(encoder_net))
            np.savez(os.path.join(save_path, "errors_unsupervised_last.npz"),
                     zip(*train_monitored), zip(*valid_monitored))

        # End training
        if patience == max_patience or epoch == num_epochs-1:
            print("   Ending training")
            # Load unsupervised best model
            if not os.path.exists(save_path + '/model_enc_unsupervised_best.npz'):
                print("No saved model to be tested and/or generate"
                      " the embedding !")
            else:
                with np.load(save_path + '/model_enc_unsupervised_best.npz',) as f:
                    param_values = [f['arr_%d' % i]
                                    for i in range(len(f.files))]
                    lasagne.layers.set_all_param_values(encoder_net, param_values)

                # Save embedding
                preds = []
                for batch in mlh.iterate_minibatches_unsup(x_train, 1,
                                                           shuffle=False):
                    preds.append(pred_feat_emb(batch))
                for batch in mlh.iterate_minibatches_unsup(x_valid, 1,
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


def main():
    parser = argparse.ArgumentParser(description="""Learn feature embedding.""")
    parser.add_argument('--dataset',
                        default='1000_genomes',
                        help='Dataset.')
    parser.add_argument('--n_hidden_u',
                        default=[100],
                        help='List of unsupervised hidden units.')
    parser.add_argument('--num_epochs',
                        '-ne',
                        type=int,
                        default=1000,
                        help="""Int to indicate the max'
                        'number of epochs.""")
    parser.add_argument('--learning_rate',
                        '-lr',
                        type=float,
                        default=.01,
                        help="""Float to indicate learning rate.""")
    parser.add_argument('--learning_rate_annealing',
                        '-lra',
                        type=float,
                        default=1.,
                        help="Float to indicate learning rate annealing rate.")
    parser.add_argument('--lmd',
                        '-l',
                        type=float,
                        default=.0001,
                        help="""Float to indicate weight decay coeff.""")
    parser.add_argument('-embedding_input',
                        type=str,
                        default='histo3x26',
                        help='The kind of input we will use for the feat. emb. nets')
    parser.add_argument('--which_fold',
                        type=int,
                        default=1,
                        help='Which fold to use for cross-validation (0-4)')
    parser.add_argument('--save_tmp',
                        default='/Tmp/'+ os.environ["USER"]+'/DietNetworks/',
                        help='Path to save results.')
    parser.add_argument('--save_perm',
                        default='/data/lisatmp4/'+ os.environ["USER"]+'/DietNetworks/',
                        help='Path to save results.')
    parser.add_argument('--dataset_path',
                        default='/data/lisatmp4/romerosa/datasets/1000_Genome_project/',
                        help='Path to dataset')


    args = parser.parse_args()

    execute(args.dataset,
            mlh.parse_int_list_arg(args.n_hidden_u),
            int(args.num_epochs),
            args.learning_rate,
            args.learning_rate_annealing,
            args.lmd,
            args.embedding_input,
            args.which_fold,
            args.save_tmp,
            args.save_perm,
            args.dataset_path)


if __name__ == '__main__':
    main()
