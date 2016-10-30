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
from lasagne.nonlinearities import softmax  # , tanh, linear
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


def monitoring(minibatches, dataset_name, val_fn, monitoring_labels,
               pred_fn=None, n_classes=2):
    """Monitor learning information."""
    monitoring_values = np.zeros(len(monitoring_labels), dtype="float32")
    all_probs = np.zeros((0, n_classes), "float32")
    all_targets = np.zeros((0), "float32")
    global_batches = 0

    for batch in minibatches:
        inputs, targets = batch

        # Update monitored values
        out = val_fn(inputs, targets)
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

    # Print supervised-specific metrics
    if pred_fn is not None:
        # Compute the confusion matrix
        all_predictions = all_probs.argmax(1)
        confusion = np.zeros((n_classes, n_classes))
        for i in range(len(all_predictions)):
            confusion[all_targets[i], all_predictions[i]] += 1

        # Print the BER (balanced error rate)
        ber = 0.5 * (confusion[0, 1] / confusion.sum(axis=1)[0] +
                     confusion[1, 0] / confusion.sum(axis=1)[1])
        print ("  {} ber:\t\t\t{:.6f}".format(dataset_name, ber))

        # Compute and print the AUC (this implementation is inefficient but
        # simple, it may be sped up if it ever becomes a bottleneck. It comes
        # from http://www.cs.ru.nl/~tomh/onderwijs/dm/dm_files/roc_auc.pdf)
        preds_for_neg_examples = all_predictions[np.argwhere(all_targets == 0)]
        preds_for_pos_examples = all_predictions[np.argwhere(all_targets == 1)]
        auc = 0.
        for neg_pred in preds_for_neg_examples:
            for pos_pred in preds_for_pos_examples:
                if pos_pred > neg_pred:
                    auc += 1.
        auc /= (len(preds_for_neg_examples) * len(preds_for_pos_examples))
        print ("  {} auc:\t\t\t{:.6f}".format(dataset_name, auc))


# Main program
def execute(dataset, feat_embedding_source,
            samp_embedding_source, num_epochs=500):
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
    save_path = '/data/lisatmp4/dejoieti/feature_selection/'
    print("Loading data")
    if samp_embedding_source is None:
        if dataset == 'genomics':
            from experiments.common.dorothea import load_data
            x_train, y_train = load_data('train', 'standard', False, 'numpy')
            x_valid, y_valid = load_data('valid', 'standard', False, 'numpy')

            # WARNING : The dorothea dataset has no test labels
            x_test = load_data('test', 'standard', False, 'numpy')
            y_test = None

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
    else:
        from experiments.common.dorothea import load_data
        f = np.load(save_path + samp_embedding_source)
        x_train = np.array(f['x_train'], dtype=np.float32)
        y_train = np.array(f['y_train'])
        x_valid = np.array(f['x_valid'], dtype=np.float32)
        y_valid = np.array(f['y_valid'])
        n_output = x_train.shape[1]

    x_test, y_test = x_valid, y_valid

    n_samples, n_feats = x_train.shape
    n_classes = y_train.max() + 1
    n_batch = 10
    save_path = '/data/lisatmp4/dejoieti/feature_selection/'

    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.ivector('targets')
    lr = theano.shared(np.float32(1e-3), 'learning_rate')

    # Build model
    print("Building model")

    if feat_embedding_source:
        feat_emb_val = np.load(save_path + feat_embedding_source).items()[0][1]
        feat_emb = theano.shared(feat_emb_val.astype('float32'), 'feat_emb')

        n_output = feat_emb_val.shape[1]

        discrim_net = InputLayer((n_batch, n_feats), input_var)
        discrim_net = DenseLayer(discrim_net, num_units=n_output, W=feat_emb)

    elif samp_embedding_source:
        discrim_net = InputLayer((n_batch, n_output), input_var)

    discrim_net = DenseLayer(discrim_net, num_units=n_output)
    discrim_net = DenseLayer(discrim_net, num_units=n_classes,
                             nonlinearity=softmax)

    # Create a loss expression for training
    print("Building and compiling training functions")

    # Expressions required for training
    prediction = lasagne.layers.get_output(discrim_net)
    loss = lasagne.objectives.categorical_crossentropy(prediction,
                                                       target_var).mean()
    params = lasagne.layers.get_all_params(discrim_net, trainable=True)

    updates = lasagne.updates.rmsprop(loss,
                                      params,
                                      learning_rate=lr)
    # updates = lasagne.updates.sgd(loss,
    #                              params,
    #                              learning_rate=lr)
    # updates = lasagne.updates.momentum(loss, params,
    #                                    learning_rate=lr, momentum=0.0)
    updates[lr] = (lr * 0.99).astype("float32")

    # Compile a function performing a training step on a mini-batch (by
    # giving the updates dictionary) and returning the corresponding
    # training loss.
    # Warnings about unused inputs are ignored because otherwise Theano might
    # complain about the targets being a useless input when doing unsupervised
    # training of the network.
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Expressions required for test
    test_prediction = lasagne.layers.get_output(discrim_net,
                                                deterministic=True)
    test_predictions_loss = lasagne.objectives.categorical_crossentropy(
        test_prediction, target_var).mean()
    test_class = T.argmax(test_prediction, axis=1)
    test_acc = T.mean(T.eq(test_class, target_var),
                      dtype=theano.config.floatX) * 100.

    val_fn = theano.function([input_var, target_var],
                             [test_predictions_loss, test_acc],
                             on_unused_input='ignore')
    pred_fn = theano.function([input_var], test_prediction,
                              on_unused_input='ignore')
    monitor_labels = ["pred. loss", "accuracy"]

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
            train_fn(inputs, targets)

        # Monitor progress
        print("Epoch {} of {}".format(epoch + 1, num_epochs))

        train_minibatches = iterate_minibatches(x_train, y_train, n_batch,
                                                shuffle=False)
        monitoring(train_minibatches, "train", val_fn,
                   monitor_labels, pred_fn, n_classes)

        # Only monitor on the validation set if training in a supervised way
        # otherwise the dimensions will not match.
        valid_minibatches = iterate_minibatches(x_valid, y_valid, n_batch,
                                                shuffle=False)
        monitoring(valid_minibatches, "valid", val_fn,
                   monitor_labels, pred_fn, n_classes)

        print("  total time:\t\t\t{:.3f}s".format(time.time() - start_time))

    # After training, we compute and print the test error (only if doing
    # supervised training or the dimensions will not match):
    test_minibatches = iterate_minibatches(x_test, y_test, n_batch,
                                           shuffle=False)
    print("Final results:")
    monitoring(test_minibatches, "test", val_fn,
               monitor_labels, pred_fn, n_classes)

    # Save network weights to a file
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if feat_embedding_source:
        np.savez(save_path+'model2.npz',
                 *lasagne.layers.get_all_param_values(discrim_net))
    else:
        np.savez(save_path+'model3.npz',
                 *lasagne.layers.get_all_param_values(discrim_net))
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
