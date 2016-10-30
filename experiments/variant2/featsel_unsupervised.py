"""
Unsupervised learning module.

This module can be used to generate an embeding of thge features.
This embedding can then be used for supervised learning.
See featsel_supervised.py
"""
from __future__ import print_function

import sys
import argparse
import time
import os

import lasagne
from lasagne.layers import DenseLayer, InputLayer
from lasagne.nonlinearities import sigmoid  # , tanh, linear
import numpy as np
import theano
import theano.tensor as T

# I had a problem with Python import path so I have to add this
sys.path.append('/data/lisatmp4/dejoieti/feature_selection')


def iterate_minibatches(inputs, targets, batchsize, axis=0, shuffle=False):
    """Generate minibatches for learning."""
    assert len(inputs) == len(targets)
    assert axis >= 0 and axis < len(inputs.shape)
    targets = targets.transpose()

    if axis == 1:
        inputs = inputs.transpose()

    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        if axis == 0:
            yield inputs[excerpt].transpose(), targets[excerpt]
        elif axis == 1:
            yield inputs[excerpt], targets


def onehot_labels(labels, min_val, max_val):
    output = np.zeros((len(labels), max_val - min_val + 1), dtype="int32")
    output[np.arange(len(labels)), labels - min_val] = 1
    return output


def monitoring(minibatches, dataset_name, val_fn, monitoring_labels,
               pred_fn=None, n_classes=2):

    monitoring_values = np.zeros(len(monitoring_labels), dtype="float32")
    all_probs = np.zeros((0, n_classes), "float32")
    all_targets = np.zeros((0), "float32")
    global_batches = 0

    for batch in minibatches:
        inputs, targets = batch

        # Update monitored values
        out = val_fn(inputs)
        monitoring_values = monitoring_values + out
        global_batches += 1

        # Update the prediction / target lists
        if pred_fn is not None:
            probs = pred_fn(inputs)
            all_probs = np.concatenate((all_probs, probs), axis=0)
            all_targets = np.concatenate((all_targets, targets), axis=0)

    # Print monitored values
    monitoring_values /= global_batches
    for (label, val) in zip(monitoring_labels, monitoring_values):
        print ("  {} {}:\t\t{:.6f}".format(dataset_name, label, val))

# Main program
def execute(dataset, n_output, num_epochs=500):
    # Load the dataset
    print("Loading data")
    if dataset == 'genomics':
        from experiments.common.dorothea import load_data
        x_train, y_train = load_data('train', 'standard', False, 'numpy')
        x_valid, y_valid = load_data('valid', 'standard', False, 'numpy')

        # There is a test set but it has no labels. For simplicity, use
        # the validation set as test set
        x_test, y_test = x_valid, y_valid

    elif dataset == 'debug':
        x_train = np.random.rand(10, 100).astype(np.float32)
        x_valid = np.random.rand(2, 100).astype(np.float32)
        x_test = np.random.rand(2, 100).astype(np.float32)
        y_train = np.random.randint(0, 2, size=10).astype('int32')
        y_valid = np.random.randint(0, 2, size=2).astype('int32')
        y_test = np.random.randint(0, 2, size=2).astype('int32')

    else:
        print("Unknown dataset")
        return

    n_samples, n_feats = x_train.shape
    n_classes = y_train.max() + 1
    n_batch = 100
    save_path = '/data/lisatmp4/dejoieti/feature_selection/'

    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    lr = theano.shared(np.float32(1e-3), 'learning_rate')

    # Build model
    print("Building model")

    encoder_net = InputLayer((n_batch, n_samples), input_var)
    encoder_net = DenseLayer(encoder_net, num_units=n_output)
    encoder_net = DenseLayer(encoder_net, num_units=n_output)
    feat_emb = lasagne.layers.get_output(encoder_net)

    decoder_net = DenseLayer(encoder_net, num_units=n_samples,
                             nonlinearity=sigmoid)
    # decoder_net = DenseLayer(encoder_net, num_units=n_samples,
    #                          nonlinearity=sigmoid)

    # Create a loss expression for training
    print("Building and compiling training functions")

    # Expressions required for training
    reconstruction = lasagne.layers.get_output(decoder_net)
    loss = lasagne.objectives.binary_crossentropy(reconstruction,
                                                  input_var).mean()
    params = lasagne.layers.get_all_params(decoder_net, trainable=True)

    updates = lasagne.updates.rmsprop(loss,
                                      params,
                                      learning_rate=lr)
    updates[lr] = (lr * 0.99).astype("float32")

    # Compile a function performing a training step on a mini-batch (by
    # giving the updates dictionary) and returning the corresponding
    # training loss.
    # Warnings about unused inputs are ignored because otherwise Theano might
    # complain about the targets being a useless input when doing unsupervised
    # training of the network.
    train_fn = theano.function([input_var], loss, updates=updates)

    # Expressions required for test
    test_reconstruction = lasagne.layers.get_output(decoder_net,
                                                    deterministic=True)
    test_reconstruction_loss = lasagne.objectives.binary_crossentropy(
        test_reconstruction, input_var).mean()

    val_fn = theano.function([input_var], [test_reconstruction_loss])
    pred_fn = None
    monitor_labels = ["recon. loss"]

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    minibatch_axis = 1
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data to updates
        # the parameters:
        start_time = time.time()
        for batch in iterate_minibatches(x_train, y_train, n_batch,
                                         minibatch_axis, shuffle=True):
            inputs, targets = batch
            train_fn(inputs)

        # Monitor progress
        print("Epoch {} of {}".format(epoch + 1, num_epochs))

        train_minibatches = iterate_minibatches(x_train, y_train, n_batch,
                                                minibatch_axis, shuffle=False)
        monitoring(train_minibatches, "train", val_fn,
                   monitor_labels, pred_fn, n_classes)

        print("  total time:\t\t\t{:.3f}s".format(time.time() - start_time))

    # Save network weights to a file
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.savez(save_path+'model1.npz',
             *lasagne.layers.get_all_param_values(decoder_net))

    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

    # Save the learnt embedding (over the training set) to a file

    # Define a function serving only to compute the feature embedding over
    # some data
    emb_fn = theano.function([input_var], feat_emb)

    # Compute the embedding over all the training data and save the result
    np.savez(save_path + "embedding_%i.npz" % n_output,
             emb_fn(x_train.transpose()))


def main():
    parser = argparse.ArgumentParser(description="""Implementation of the
                                     feature embedding unsupervised v2""")
    parser.add_argument('-dataset',
                        default='debug',
                        help='Dataset.')
    parser.add_argument('-n_output',
                        default=100,
                        help='Output dimension.')

    parser.add_argument('--num_epochs',
                        '-ne',
                        type=int,
                        default=5,
                        help="""Optional. Int to indicate the max'
                        'number of epochs.""")

    args = parser.parse_args()

    execute(args.dataset, int(args.n_output), int(args.num_epochs))


if __name__ == '__main__':
    main()
