"""
Module for supervised learning.

This module should only be used when already having an embedding for the
features or an embedding of the data.
For getting an embedding of the features, see for instance
featsel_unsupervised.py
For getting an embedding of the data, see for instance
benchmark/pca.py or benchmark/kmeans.py

"""
from __future__ import print_function

import sys
import argparse
import time
import os

import lasagne
from lasagne.layers import DenseLayer, InputLayer
import numpy as np
import theano
import theano.tensor as T

sys.path.append('/data/lisatmp4/dejoieti/feature_selection')


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """Generate the minibatches for learning."""
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        yield inputs[excerpt], targets[excerpt]


def onehot_labels(labels, min_val, max_val):
    output = np.zeros((len(labels), max_val - min_val + 1), dtype="int32")
    output[np.arange(len(labels)), labels - min_val] = 1
    return output


def generate_test_predictions(minibatches, pred_fn):

    # Obtain the predictions over all the examples
    all_predictions = np.zeros((0), "int32")
    all_probabilities = np.zeros((0), "float32")
    for batch in minibatches:
        inputs, _ = batch

        probs = pred_fn(inputs)
        all_probabilities = np.concatenate((all_probabilities, probs[:, 1]),
                                           axis=0)

        predictions = probs.argmax(axis=1)
        all_predictions = np.concatenate((all_predictions, predictions),
                                         axis=0)

    # Write the predictions to a text file
    filename_pred = "test_preds_" + time.strftime("%Y-%M-%d_%T") + ".txt"
    with open(filename_pred, "w") as f:
        f.write(",".join([str(p) for p in all_predictions]))

    # Also write the probabilities of the positive class to a text file
    filename_prob = "test_probs_" + time.strftime("%Y-%M-%d_%T") + ".txt"
    with open(filename_prob, "w") as f:
        f.write(",".join([str(p) for p in all_probabilities]))


def monitoring(minibatches, dataset_name, val_fn, monitoring_labels):

    monitoring_values = np.zeros(len(monitoring_labels), dtype="float32")
    monitoring_dict = {}
    global_batches = 0

    for batch in minibatches:
        inputs, targets = batch

        # Update monitored values
        out = val_fn(inputs, targets.astype("float32"))
        monitoring_values = monitoring_values + out[0]
        global_batches += 1

    # Print monitored values
    monitoring_values /= global_batches
    for (label, val) in zip(monitoring_labels, monitoring_values):
        print ("  {}\t{}:\t\t{:.6f}".format(dataset_name, label, val))
        monitoring_dict[label] = val

    return monitoring_dict


# Main program
def execute(samp_embedding_source, num_epochs=500,
            lr_value=1e-5, split_valid=.15,
            save_path='/data/lisatmp4/dejoieti/feature_selection/'):
    """
    Execute a supervised learning.

    This function execute a suprevised learning using a embedding
    of the features or directly an embedding of the data. You must
    provide either one of them.

    :param dataset: the dataset you want to use. genomics by default
    :param feat_embedding_source: if you want to choose a feature embedding,
    you should provide the path of the npz file from save_path
    ('/data/lisatmp4/dejoieti/feature_selection/')
    :param sample_embedding_source: if you want to choose directly an embedding
    of the data, you should provide the path of the npz file from save_path
    ('/data/lisatmp4/dejoieti/feature_selection/')
    """

    # Load the dataset
    print("Loading data")
    f = np.load(save_path + samp_embedding_source)

    print (f.files)
    x_train = np.array(f['x_train_supervised'], dtype=np.float32)
    y_train = np.array(f['y_train_supervised'])
    x_test = np.array(f['x_test_supervised'], dtype=np.float32)
    y_test = np.array(f['y_test_supervised'])

    # x_train = np.array(f['x_train'], dtype=np.float32)
    # y_train = np.array(f['y_train'])
    # x_test = np.array(f['x_valid'], dtype=np.float32)
    # y_test = np.array(f['y_valid'])

    n_data = x_train.shape[0] + x_test.shape[0]
    end_train = int(split_valid*n_data)
    x_valid = x_train[end_train:]
    y_valid = y_train[end_train:]
    x_train = x_train[:end_train]
    y_train = y_train[:end_train]

    n_samples, n_feats = x_train.shape
    n_batch = 10

    print("Building model")
    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.vector('targets',  dtype="float32")
    lr = theano.shared(np.float32(lr_value), 'learning_rate')

    # Build model
    regression_net = InputLayer((n_batch, n_feats), input_var)
    regression_net = DenseLayer(
        regression_net, num_units=1,
        nonlinearity=lasagne.nonlinearities.very_leaky_rectify)

    # Create a loss expression for training
    print("Building and compiling training functions")

    # Expressions required for training
    prediction = lasagne.layers.get_output(regression_net)[:, 0]
    loss = lasagne.objectives.squared_error(prediction, target_var).mean()
    params = lasagne.layers.get_all_params(regression_net, trainable=True)

    updates = lasagne.updates.rmsprop(loss,
                                      params,
                                      learning_rate=lr)
    # updates = lasagne.updates.sgd(loss,
    #                               params,
    #                               learning_rate=lr)
    # updates = lasagne.updates.momentum(loss, params,
    #                                    learning_rate=lr, momentum=0.0)
    updates[lr] = (lr * 0.99).astype("float32")

    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Expressions required for test
    test_prediction = \
        lasagne.layers.get_output(regression_net,
                                  deterministic=True)[:, 0]
    test_predictions_loss = lasagne.objectives.squared_error(
        test_prediction, target_var).mean()

    val_fn = theano.function([input_var, target_var],
                             [test_predictions_loss,
                              test_prediction],
                             on_unused_input='ignore')
    monitor_labels = ["pred. loss"]

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data to updates
        # the parameters:
        start_time = time.time()
        for batch in iterate_minibatches(x_train, y_train, n_batch,
                                         shuffle=True):
            inputs, targets = batch
            train_fn(inputs, targets.astype("float32"))

        # Monitor progress
        print("Epoch {} of {}".format(epoch + 1, num_epochs))

        train_minibatches = iterate_minibatches(x_train, y_train, n_batch,
                                                shuffle=False)
        monitoring(train_minibatches, "train", val_fn,
                   monitor_labels)

        # Only monitor on the validation set if training in a supervised way
        # otherwise the dimensions will not match.
        valid_minibatches = iterate_minibatches(x_valid, y_valid, n_batch,
                                                shuffle=False)
        monitoring(valid_minibatches, "valid", val_fn,
                   monitor_labels)

        print("  total time:\t\t\t{:.3f}s".format(time.time() - start_time))

    # After training, we compute and print the test error (only if doing
    # supervised training or the dimensions will not match):
    test_minibatches = iterate_minibatches(x_test, y_test, n_batch,
                                           shuffle=False)
    train_minibatches = iterate_minibatches(x_train, y_train, n_batch,
                                            shuffle=False)
    valid_minibatches = iterate_minibatches(x_valid, y_valid, n_batch,
                                            shuffle=False)

    print("Final results:")

    #import pdb
    #pdb.set_trace()
    test_mon = monitoring(test_minibatches, "test", val_fn,
                          monitor_labels)
    valid_mon = monitoring(valid_minibatches, "valid", val_fn,
                           monitor_labels)
    train_mon = monitoring(train_minibatches, "train", val_fn,
                           monitor_labels)

    #import pdb
    #pdb.set_trace()
    save_path = os.path.join(save_path,"results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print ("save_path: {}".format(save_path))

    np.savez(save_path+"errors_" + str(lr_value) + "_"+samp_embedding_source,
             test_err=test_mon["pred. loss"],
             valid_err=valid_mon["pred. loss"],
             train_err=train_mon["pred. loss"])

    # Save network weights to a file


    np.savez(save_path+"/regression_" + str(lr_value) + "_" +samp_embedding_source,
             *lasagne.layers.get_all_param_values(regression_net))
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


def main():
    """Run execute with the accurate arguments."""
    parser = argparse.ArgumentParser(description="""Implementation of the
                                     feature selection v2""")
    parser.add_argument('-dataset',
                        default='genomics',
                        help='Dataset.')
    parser.add_argument('-feat_embedding_source',
                        '-fes',
                        default=None,
                        help='Source for the feature embedding. Either' +
                             '"predicted" or the name of a file from which' +
                             'to load a learned embedding')
    parser.add_argument('-sample_embedding_source',
                        '-ses',
                        default=None,
                        help='Source for the sample embedding. Either' +
                             '"predicted" or the name of a file from which' +
                             'to load a learned embedding')
    parser.add_argument('--num_epochs',
                        '-ne',
                        type=int,
                        default=5,
                        help="""Optional. Int to indicate the max'
                        'number of epochs.""")

    args = parser.parse_args()

    execute(args.dataset, args.feat_embedding_source,
            args.sample_embedding_source, int(args.num_epochs))


if __name__ == '__main__':
    main()
