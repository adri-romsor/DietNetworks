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


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    targets = targets.transpose()
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt].transpose(), targets[excerpt]


def onehot_labels(labels, min_val, max_val):
    output = np.zeros((len(labels), max_val - min_val + 1), dtype="int32")
    output[np.arange(len(labels)), labels - min_val] = 1
    return output


def print_monitoring(dataset, loss, p_loss, r_loss, acc, num_batches):
    print("  {} total loss:\t\t{:.6f}".format(dataset, loss / num_batches))
    print("  {} pred. loss:\t\t{:.6f}".format(dataset, p_loss / num_batches))
    print("  {} recon. loss:\t\t{:.6f}".format(dataset, r_loss / num_batches))
    print("  {} accuracy:\t\t{:.2f}%".format(dataset, acc / num_batches * 100))


# Main program
def execute(training, dataset, n_output, num_epochs=500):
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
    network1 = InputLayer((n_feats, n_batch), input_var)
    network1 = DenseLayer(network1, num_units=n_output)
    feat_emb = lasagne.layers.get_output(network1)
    network3 = DenseLayer(network1, num_units=n_batch, nonlinearity=sigmoid)

    network2 = InputLayer((n_batch, n_feats), input_var.transpose())
    network2 = DenseLayer(network2, num_units=10, W=feat_emb)
    network2 = DenseLayer(network2, num_units=n_classes, nonlinearity=softmax)

    # Create a loss expression for training
    print("Building and compiling training functions")
    # Expressions required for training

    reconstruction = lasagne.layers.get_output(network3)
    reconstruction_loss = lasagne.objectives.binary_crossentropy(reconstruction, input_var).mean()
    prediction = lasagne.layers.get_output(network2)
    prediction_loss = lasagne.objectives.categorical_crossentropy(prediction, target_var).mean()

    params_sup = lasagne.layers.get_all_params(network2, trainable=True)
    params_unsup = lasagne.layers.get_all_params(network3, trainable=True)

    if training == "supervised":
        loss = prediction_loss
        params = params_sup
    elif training == "semi_supervised":
        loss = reconstruction_loss + prediction_loss
        params = params_sup + params_unsup
    elif training == "unsupervised":
        loss = reconstruction_loss
        parama = params_unsup

    updates = lasagne.updates.rmsprop(loss,
                                      params,
                                      learning_rate=lr)
    #updates = lasagne.updates.momentum(loss, params,
    #                                   learning_rate=lr, momentum=0.0)
    updates[lr] = (lr * 0.99).astype("float32")

    # Compile a function performing a training step on a mini-batch (by
    # giving the updates dictionary) and returning the corresponding
    # training loss:
    train_fn = theano.function([input_var, target_var], loss,
                               updates=updates)

    # Expressions required for test
    test_reconstruction = lasagne.layers.get_output(network3, deterministic=True)
    test_reconstruction_loss = lasagne.objectives.binary_crossentropy(test_reconstruction,
                                                                      input_var).mean()
    test_prediction = lasagne.layers.get_output(network2, deterministic=True)
    test_predictions_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                        target_var).mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    val_fn = theano.function([input_var, target_var],
                             [test_predictions_loss, test_reconstruction_loss,
                              test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data to updates
        # the parameters:
        start_time = time.time()
        for batch in iterate_minibatches(x_train, y_train,
                                         n_batch, shuffle=True):
            inputs, targets = batch
            train_fn(inputs, targets)

        # A second pass over the training data for monitoring :
        train_loss = 0
        train_p_loss = 0
        train_r_loss = 0
        train_acc = 0
        train_batches = 0
        for batch in iterate_minibatches(x_train, y_train,
                                         n_batch, shuffle=True):
            inputs, targets = batch
            p_loss, r_loss, acc = val_fn(inputs, targets)
            train_loss += (r_loss + p_loss)
            train_p_loss += p_loss
            train_r_loss += r_loss
            train_acc += acc
            train_batches += 1

        # And a full pass over the validation data for monitoring:
        val_loss = 0
        val_p_loss = 0
        val_r_loss = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(x_valid, y_valid,
                                         n_batch, shuffle=False):
            inputs, targets = batch
            p_loss, r_loss, acc = val_fn(inputs, targets)
            val_loss += (r_loss + p_loss)
            val_p_loss += p_loss
            val_r_loss += r_loss
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))

        print_monitoring("train", train_loss, train_p_loss, train_r_loss,
                         train_acc, train_batches)
        print_monitoring("valid", val_loss, val_p_loss, val_r_loss,
                         val_acc, val_batches)

    # After training, we compute and print the test error:
    test_p_loss = 0
    test_r_loss = 0
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(x_test, y_test,
                                     n_batch, shuffle=False):
        inputs, targets = batch
        p_loss, r_loss, acc = val_fn(inputs, targets)
        test_p_loss += p_loss
        test_r_loss += r_loss
        test_acc += acc
        test_batches += 1

    # Print metrics
    print("Final results:")
    print("  test pred loss:\t\t\t{:.6f}".format(test_p_loss / test_batches))
    print("  test recon loss:\t\t\t{:.6f}".format(test_r_loss / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Save network weights to a file
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.savez(save_path+'model1.npz',
             *lasagne.layers.get_all_param_values(network1))
    np.savez(save_path+'model2.npz',
             *lasagne.layers.get_all_param_values(network2))


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
    parser.add_argument('--num_epochs',
                        '-ne',
                        type=int,
                        default=500,
                        help="""Optional. Int to indicate the max'
                        'number of epochs.""")

    args = parser.parse_args()

    execute(args.training, args.dataset, int(args.n_output),
            int(args.num_epochs))


if __name__ == '__main__':
    main()
