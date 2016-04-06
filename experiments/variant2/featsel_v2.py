from __future__ import print_function
import argparse
import time
import os

import lasagne
from lasagne.layers import DenseLayer, InputLayer
from lasagne.nonlinearities import sigmoid, softmax, tanh, linear
import numpy as np
import theano
import theano.tensor as T


def iterate_minibatches(inputs, targets, batchsize, axis=0, shuffle=False):
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


def monitoring(val_fn, dataset_name, minibatches):
    global_loss = 0
    global_p_loss = 0
    global_r_loss = 0
    global_acc = 0
    global_batches = 0

    for batch in minibatches:
        inputs, targets = batch
        p_loss, r_loss, acc = val_fn(inputs, targets)
        global_loss += (r_loss + p_loss)
        global_p_loss += p_loss
        global_r_loss += r_loss
        global_acc += acc
        global_batches += 1

    global_loss /= global_batches
    global_p_loss /= global_batches
    global_r_loss /= global_batches
    global_acc = global_acc / global_batches * 100

    print("  {} total loss:\t\t{:.6f}".format(dataset_name, global_loss))
    print("  {} pred. loss:\t\t{:.6f}".format(dataset_name, global_p_loss))
    print("  {} recon. loss:\t\t{:.6f}".format(dataset_name, global_r_loss))
    print("  {} accuracy:\t\t{:.2f}%".format(dataset_name, global_acc))


def print_monitoring(dataset, loss, p_loss, r_loss, acc, num_batches):
    print("  {} total loss:\t\t{:.6f}".format(dataset, loss / num_batches))
    print("  {} pred. loss:\t\t{:.6f}".format(dataset, p_loss / num_batches))
    print("  {} recon. loss:\t\t{:.6f}".format(dataset, r_loss / num_batches))
    print("  {} accuracy:\t\t{:.2f}%".format(dataset, acc / num_batches * 100))


# Main program
def execute(training, dataset, n_output, embedding_source, num_epochs=500):
    # Load the dataset
    print("Loading data")
    if dataset == 'genomics':
        from feature_selection.experiments.common.dorothea import load_data
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
    n_batch = 20
    save_path = '/data/lisatmp4/carriepl/FeatureSelection/'

    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.ivector('targets')
    lr = theano.shared(np.float32(1e-2), 'learning_rate')

    # Build model
    print("Building model")

    if embedding_source == "predicted":
        encoder_net = InputLayer((n_batch, n_samples), input_var)
        encoder_net = DenseLayer(encoder_net, num_units=n_output)
        encoder_net = DenseLayer(encoder_net, num_units=n_output)
        feat_emb = lasagne.layers.get_output(encoder_net)
    else:
        feat_emb = theano.shared(np.load(save_path + embedding_source),
                                 'feat_emb')

    decoder_net = DenseLayer(encoder_net, num_units=n_output)
    decoder_net = DenseLayer(decoder_net, num_units=n_samples, nonlinearity=sigmoid)

    discrim_net = InputLayer((n_batch, n_feats), input_var.transpose())
    discrim_net = DenseLayer(discrim_net, num_units=n_output, W=feat_emb)
    discrim_net = DenseLayer(discrim_net, num_units=n_classes, nonlinearity=softmax)

    # Create a loss expression for training
    print("Building and compiling training functions")
    # Expressions required for training

    reconstruction = lasagne.layers.get_output(decoder_net)
    reconstruction_loss = lasagne.objectives.binary_crossentropy(reconstruction, input_var).mean()
    prediction = lasagne.layers.get_output(discrim_net)
    prediction_loss = lasagne.objectives.categorical_crossentropy(prediction, target_var).mean()

    params_sup = lasagne.layers.get_all_params(discrim_net, trainable=True)
    params_unsup = lasagne.layers.get_all_params(decoder_net, trainable=True)

    if training == "supervised":
        loss = prediction_loss
        params = params_sup
    elif training == "semi_supervised":
        loss = reconstruction_loss + prediction_loss
        params = params_sup + params_unsup
    elif training == "unsupervised":
        loss = reconstruction_loss
        params = params_unsup

    updates = lasagne.updates.rmsprop(loss,
                                      params,
                                      learning_rate=lr)
    #updates = lasagne.updates.momentum(loss, params,
    #                                   learning_rate=lr, momentum=0.0)
    updates[lr] = (lr * 0.99).astype("float32")

    # Compile a function performing a training step on a mini-batch (by
    # giving the updates dictionary) and returning the corresponding
    # training loss.
    # Warnings about unused inputs are ignored because otherwise Theano might
    # complain about the targets being a useless input when doing unsupervised
    # training of the network.
    train_fn = theano.function([input_var, target_var], loss,
                               updates=updates,
                               on_unused_input='ignore')

    # Expressions required for test
    test_reconstruction = lasagne.layers.get_output(decoder_net, deterministic=True)
    test_reconstruction_loss = lasagne.objectives.binary_crossentropy(test_reconstruction,
                                                                      input_var).mean()
    test_prediction = lasagne.layers.get_output(discrim_net, deterministic=True)
    test_predictions_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                        target_var).mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    val_fn = theano.function([input_var, target_var],
                             [test_predictions_loss, test_reconstruction_loss,
                              test_acc])

    # Define a function serving only to compute the feature embedding over
    # some data
    emb_fn = theano.function([input_var], feat_emb)

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    minibatch_axis = int(training == "unsupervised")
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data to updates
        # the parameters:
        start_time = time.time()
        for batch in iterate_minibatches(x_train, y_train, n_batch,
                                         minibatch_axis, shuffle=True):
            inputs, targets = batch
            train_fn(inputs, targets)

        # Monitor progress
        print("Epoch {} of {}".format(epoch + 1, num_epochs))

        train_minibatches = iterate_minibatches(x_train, y_train, n_batch,
                                                minibatch_axis, shuffle=False)
        monitoring(val_fn, "train", train_minibatches)

        # Only monitor on the validation set if training in a supervised way
        # otherwise the dimensions will not match.
        if training == "supervised":
            valid_minibatches = iterate_minibatches(x_valid, y_valid, n_batch,
                                                    minibatch_axis,
                                                    shuffle=False)
            monitoring(val_fn, "valid", valid_minibatches)

        print("  total time:\t\t\t{:.3f}s".format(time.time() - start_time))

    # After training, we compute and print the test error (only if doing
    # supervised training or the dimensions will not match):
    if training == "supervised":
        test_minibatches = iterate_minibatches(x_test, y_test, n_batch,
                                               minibatch_axis, shuffle=False)
        print("Final results:")
        monitoring(val_fn, "test", test_minibatches)

    # Save network weights to a file
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.savez(save_path+'model1.npz',
             *lasagne.layers.get_all_param_values(decoder_net))
    np.savez(save_path+'model2.npz',
             *lasagne.layers.get_all_param_values(discrim_net))


    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


def main():
    parser = argparse.ArgumentParser(description="""Implementation of the
                                     feature selection v2""")
    parser.add_argument('training',
                        default='supervised',
                        help='Type of training.')
    parser.add_argument('dataset',
                        default='debug',
                        help='Dataset.')
    parser.add_argument('n_output',
                        default=100,
                        help='Output dimension.')
    parser.add_argument('embedding_source',
                        default="predicted",
                        help='Source for the feature embedding. Either' +
                             '"predicted" or the name of a file from which' +
                             'to load a learned embedding')
    parser.add_argument('--num_epochs',
                        '-ne',
                        type=int,
                        default=5,
                        help="""Optional. Int to indicate the max'
                        'number of epochs.""")

    args = parser.parse_args()

    execute(args.training, args.dataset, int(args.n_output),
            args.embedding_source, int(args.num_epochs))


if __name__ == '__main__':
    main()
