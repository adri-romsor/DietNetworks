import sys
sys.path.append('/data/lisatmp4/dejoieti/')
import argparse
import time
import os

from feature_selection.config import path_dorothea
import lasagne
from lasagne.layers import DenseLayer, InputLayer
from lasagne.nonlinearities import sigmoid, softmax
import numpy as np
import theano
import theano.tensor as T
from feature_selection.experiments.variant2.featsel_v2 import monitoring

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
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


def execute(dataset, feat_embedding_source, samp_embedding_source, num_epochs=500):
    # Load the dataset
    print("Loading data")
    save_path = '/data/lisatmp4/dejoieti/feature_selection/'

    if samp_embedding_source == None:
        from feature_selection.experiments.common.dorothea import load_data
        x_train, y_train = load_data('train', 'standard', False, 'numpy')
        x_valid, y_valid = load_data('valid', 'standard', False, 'numpy')
    else:
        from feature_selection.experiments.common.dorothea import load_data
        f = np.load(save_path + samp_embedding_source)
        x_train = np.array(f['x_train'], dtype=np.float32)
        y_train = np.array(f['y_train'])
        x_valid = np.array(f['x_valid'], dtype=np.float32)
        y_valid = np.array(f['y_valid'])

    # There is a test set but it has no labels. For simplicity, use
    # the validation set as test set
    x_test, y_test = x_valid, y_valid

    n_samples, n_feats = x_train.shape
    print x_train.shape
    n_classes = y_train.max() + 1
    n_batch = 100

    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.ivector('targets')
    lr = theano.shared(np.float32(1e-3), 'learning_rate')

    # Build model
    print("Building model")

    discrim_net = InputLayer((n_batch, n_feats), input_var)
    if samp_embedding_source == None:
        feat_emb_val = np.load(save_path + feat_embedding_source).items()[0][1]
        feat_emb = theano.shared(feat_emb_val, 'feat_emb')
        discrim_net =
    else:
        discrim_net = DenseLayer(discrim_net, num_units=n_feats)
        discrim_net = DenseLayer(discrim_net, num_units=n_classes,
                                 nonlinearity=softmax)

     # Create a loss expression for training
    print("Building and compiling training functions")

    prediction = lasagne.layers.get_output(discrim_net)
    loss = lasagne.objectives.categorical_crossentropy(prediction,
                                                        target_var).mean()
    params = lasagne.layers.get_all_params(discrim_net, trainable=True)

    updates = lasagne.updates.rmsprop(loss,
                                      params,
                                      learning_rate=lr)
    updates[lr] = (lr * 0.99).astype("float32")

    train_fn = theano.function([input_var, target_var], loss,
                               updates=updates)

    # Expressions required for test
    test_prediction = lasagne.layers.get_output(discrim_net,
                                                deterministic=True)
    test_predictions_loss = lasagne.objectives.categorical_crossentropy(
        test_prediction, target_var).mean()
    test_class = T.argmax(test_prediction, axis=1)
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX) * 100.

    val_fn = theano.function([input_var, target_var],
                             [test_predictions_loss, test_acc])
    pred_fn = theano.function([input_var], test_prediction)
    monitor_labels = ["pred. loss", "accuracy"]

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:

    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data to updates
        # the parameters:
        start_time = time.time()
        for batch in iterate_minibatches(x_train, y_train, n_batch, shuffle=True):
            inputs, targets = batch
            train_fn(inputs, targets)

        # Monitor progress
        print("Epoch {} of {}".format(epoch + 1, num_epochs))

        train_minibatches = iterate_minibatches(x_train, y_train, n_batch, shuffle=False)
        monitoring(train_minibatches, "train", val_fn,
                   monitor_labels, pred_fn, n_classes)

        # Monitor on the validation set
        valid_minibatches = iterate_minibatches(x_valid, y_valid, n_batch, shuffle=False)
        monitoring(valid_minibatches, "valid", val_fn,
                   monitor_labels, pred_fn, n_classes)

    # After training, we compute and print the test error
    test_minibatches = iterate_minibatches(x_test, y_test, n_batch, shuffle=False)
    print("Final results:")
    monitoring(test_minibatches, "test", val_fn,
               monitor_labels, pred_fn, n_classes)

    np.savez(save_path+'model2.npz',
             *lasagne.layers.get_all_param_values(discrim_net))

def main():
    parser = argparse.ArgumentParser(description="""Implementation of the
                                     feature selection v2""")
    parser.add_argument('-dataset',
                        default='dorothea',
                        help='Dataset.')
    parser.add_argument('-sample_embedding_source',
                        '-ses',
                        default=None,
                        help='Source for the sample embedding. Either' +
                             '"predicted" or the name of a file from which' +
                             'to load a learned embedding')
    parser.add_argument('-feature_embedding_source',
                        '-fes',
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

    execute(args.dataset, args.feat_embedding_source, args.sample_embedding_source, args.num_epochs)


if __name__ == '__main__':
    main()
