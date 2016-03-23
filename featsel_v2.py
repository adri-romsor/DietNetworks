from __future__ import print_function
import argparse
import time
import os

import lasagne
from lasagne.layers import DenseLayer, InputLayer
from lasagne.nonlinearities import softmax
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


# Main program
def execute(training, dataset, n_output, num_epochs=500):
    # Load the dataset
    print("Loading data")
    if dataset == 'genomics':
        #    x_train, y_train, x_valid, \
        #            y_valid, x_test, y_test = data_genomics.load_data()
        pass
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
    n_batch = 1
    save_path = '/data/lisatmp4/romerosa/FeatureSelection/'

    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.ivector('targets')

    # Build model
    print("Building model")
    network1 = InputLayer((n_feats, n_batch), input_var)
    network1 = DenseLayer(network1, num_units=n_output)
    feat_emb = lasagne.layers.get_output(network1)

    network2 = InputLayer((n_batch, n_feats), input_var.transpose())
    network2 = DenseLayer(network2, num_units=n_output, W=feat_emb)
    network2 = DenseLayer(network2, num_units=n_classes, nonlinearity=softmax)

    # Create a loss expression for training
    print("Building and compiling training functions")
    # Expressions required for training
    prediction = lasagne.layers.get_output(network2)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    params1 = lasagne.layers.get_all_params(network1, trainable=True)
    params2 = lasagne.layers.get_all_params(network2, trainable=True)
    updates = lasagne.updates.rmsprop(loss,
                                      params1+params2,
                                      learning_rate=0.00001)

    # Compile a function performing a training step on a mini-batch (by
    # giving the updates dictionary) and returning the corresponding
    # training loss:
    train_fn = theano.function([input_var, target_var], loss,
                               updates=updates)

    # Expressions required for test
    test_prediction = lasagne.layers.get_output(network2, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(x_train, y_train,
                                         n_batch, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(x_valid, y_valid,
                                         n_batch, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err /
                                                  train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err /
                                                    val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(x_test, y_test,
                                     n_batch, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1

    # Print metrics
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
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
