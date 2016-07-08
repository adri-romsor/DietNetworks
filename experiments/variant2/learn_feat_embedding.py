from __future__ import print_function
import argparse
import time
import os

import lasagne
from lasagne.layers import DenseLayer, InputLayer
from lasagne.nonlinearities import sigmoid, softmax  # , tanh, linear
import numpy as np
import theano
import theano.tensor as T

def iterate_minibatches(x, batch_size, shuffle=True):
    if shuffle:
        np.random.shuffle(x)
    for i in range(0, x.shape[0]-batch_size, batch_size):
        yield x[i:i+batch_size]


def monitoring(minibatches, which_set, error_fn, monitoring_labels):

    monitoring_values = np.zeros(len(monitoring_labels), dtype="float32")
    global_batches = 0

    for batch in minibatches:
        # Update monitored values
        out = error_fn(batch)

        monitoring_values = monitoring_values + out
        global_batches += 1

    # Print monitored values
    monitoring_values /= global_batches
    for (label, val) in zip(monitoring_labels, monitoring_values):
        print ("  {} {}:\t\t{:.6f}".format(which_set, label, val))

    return monitoring_values


# Main program
def execute(dataset, n_hidden_u, unsupervised=[], num_epochs=500,
            learning_rate=.001,
            save_path='/Tmp/romerosa/feature_selection/newmodel/'):

    # Load the dataset
    print("Loading data")
    if dataset == 'protein_binding':
        from experiments.common.protein_loader import load_data
        x, _ = load_data()
    elif dataset == 'dorothea':
        from feature_selection.experiments.common.dorothea import load_data

    elif dataset == 'opensnp':
        from feature_selection import aggregate_dataset

        # This splits the data into [0.6, 0.2, 0.2] for the supervised examples
        # and puts half of the unsupervised examples in the training set
        # (because otherwise it would be way to expensive memory-wise)
        # Shuffle x_train and y_train together

    else:
        print("Unknown dataset")
        return

    split = 0.75
    if split:
        n_train = int(split*x.shape[0])
        x_train = x[:n_train]
        x_valid = x[n_train:]
    else:
        x_train = x

    # np.random.seed(0)
    # indices = np.arange(x_train.shape[0])
    # np.random.shuffle(indices)
    #
    # x_train = x_train[indices]
    # y_train = y_train[indices]
    #
    # # Standardize the dtype
    # x_train = x_train.astype("float32")
    # x_valid = x_valid.astype("float32")

    # Extract required information from data
    n_row, n_col = x_train.shape

    # Set some variables
    batch_size = 100

    # Preparing folder to save stuff
    save_path = save_path + dataset + "/"
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
    encoder_net = InputLayer((batch_size, n_col), input_var)

    for out in n_hidden_u:
        encoder_net = DenseLayer(encoder_net, num_units=out,
                                 nonlinearity=sigmoid)
    feat_emb = lasagne.layers.get_output(encoder_net)
    pred_feat_emb = theano.function([input_var], feat_emb)

    if 'autoencoder' in unsupervised:
        decoder_net = encoder_net
        for i in range(len(n_hidden_u)-2, -1, -1):
            decoder_net = DenseLayer(decoder_net, num_units=n_hidden_u[i],
                                     nonlinearity=sigmoid)
        decoder_net = DenseLayer(decoder_net, num_units=n_col,
                                 nonlinearity=sigmoid)
        reconstruction = lasagne.layers.get_output(decoder_net)
    if 'epls' in unsupervised:
        raise NotImplementedError

    print("Building and compiling training functions")
    # Some variables
    loss_auto = 0
    loss_auto_det = 0
    loss_epls = 0
    loss_epls_det = 0
    params = []

    # Build and compile training functions
    if "autoencoder" in unsupervised:
        # Unsupervised reconstruction functions
        reconstruction = lasagne.layers.get_output(decoder_net)
        reconstruction_det = lasagne.layers.get_output(decoder_net,
                                                       deterministic=True)

        loss_auto = lasagne.objectives.squared_error(
            reconstruction,
            input_var).mean()
        loss_auto_det = lasagne.objectives.squared_error(
            reconstruction_det,
            input_var).mean()

        params += lasagne.layers.get_all_params(decoder_net, trainable=True)
    if "epls" in unsupervised:
        # Unsupervised epls functions
        raise NotImplementedError

    # Combine losses
    loss = loss_auto + loss_epls
    loss_det = loss_auto_det + loss_epls_det

    # Compute network updates
    updates = lasagne.updates.rmsprop(loss,
                                      params,
                                      learning_rate=lr)
    # updates = lasagne.updates.sgd(loss,
    #                              params,
    #                              learning_rate=lr)
    # updates = lasagne.updates.momentum(loss, params,
    #                                    learning_rate=lr, momentum=0.0)

    # Compile training function
    train_fn = theano.function([input_var], loss,
                               updates=updates,
                               on_unused_input='ignore')

    # Expressions required for test
    monitor_labels = ['total_loss_det']
    val_outputs = [loss_det]

    if "autoencoder" in unsupervised:
        # Unsupervised reconstruction functions
        val_outputs += [loss_auto_det]
        monitor_labels += ["recon. loss"]

    if "epls" in unsupervised:
        # Unsupervised epls functions
        raise NotImplementedError

    # Compile validation function
    val_fn = theano.function([input_var],
                             val_outputs)

    # Finally, launch the training loop.
    print("Starting training...")

    # Some variables
    max_patience = 100
    patience = 0

    train_loss = []
    valid_loss = []
    valid_loss_auto = []
    valid_loss_epls = []

    nb_minibatches = n_row/batch_size
    start_training = time.time()
    for epoch in range(num_epochs):
        start_time = time.time()
        print("Epoch {} of {}".format(epoch+1, num_epochs))

        loss_epoch = 0

        # Train pass
        for batch in iterate_minibatches(x_train, batch_size, shuffle=True):
            loss_epoch += train_fn(batch)

        loss_epoch /= nb_minibatches
        train_loss += [loss_epoch]

        # Validation pass
        valid_minibatches = iterate_minibatches(x_valid, batch_size,
                                                shuffle=False)

        valid_err = monitoring(valid_minibatches, "valid", val_fn,
                               monitor_labels)

        valid_loss += [valid_err[0]]
        pos = 1
        if 'autoencoder' in unsupervised:
            valid_loss_auto += [valid_err[pos]]
            pos += 1
        if 'epls' in unsupervised:
            valid_loss_epls += [valid_err[pos]]

        # Eearly stopping
        if epoch == 0:
            best_valid = valid_loss[epoch]
        elif valid_loss[epoch] < best_valid:
            best_valid = valid_loss[epoch]
            patience = 0

            # Save stuff
            np.savez(save_path+'model_unsupervised.npz',
                     *lasagne.layers.get_all_param_values(encoder_net))
            np.savez(save_path + "errors_unsupervised.npz",
                     valid_loss_auto, valid_loss_epls)
        else:
            patience += 1

        # End training
        if patience == max_patience or epoch == num_epochs-1:
            print("   Ending training")
            # Load unsupervised best model
            with np.load(save_path + 'model_unsupervised.npz',) as f:
                param_values = [f['arr_%d' % i]
                                for i in range(len(f.files))]
                nlayers = len(lasagne.layers.get_all_params(encoder_net))
                lasagne.layers.set_all_param_values(encoder_net,
                                                    param_values[:nlayers])

                # Save embedding
                for batch in iterate_minibatches(x, n_row, shuffle=False):
                    pred = pred_feat_emb(batch)
                np.savez(save_path+'feature_embedding.npz', pred)

            # Stop
            print(" epoch time:\t\t\t{:.3f}s".format(time.time() - start_time))
            break

        print("  epoch time:\t\t\t{:.3f}s".format(time.time() - start_time))

    # Print all final errors for train, validation and test
    print("Training time:\t\t\t{:.3f}s".format(time.time() - start_training))


def main():
    parser = argparse.ArgumentParser(description="""Implementation of the
                                     feature selection v4""")
    parser.add_argument('--dataset',
                        default='protein_binding',
                        help='Dataset.')
    parser.add_argument('--n_hidden_u',
                        default=[100],
                        help='List of unsupervised hidden units.')
    parser.add_argument('--unsupervised',
                        default=['autoencoder'],
                        help='Add unsupervised part of the network:' +
                             'list containinge autoencoder and/or epls' +
                             'or []')
    parser.add_argument('--num_epochs',
                        '-ne',
                        type=int,
                        default=5,
                        help="""Int to indicate the max'
                        'number of epochs.""")
    parser.add_argument('--learning_rate',
                        '-lr',
                        type=float,
                        default=.0001,
                        help="""Float to indicate learning rate.""")

    parser.add_argument('--save',
                        default='/Tmp/romerosa/feature_selection/' +
                                'newmodel/',
                        help='Path to save results.')

    args = parser.parse_args()

    if args.unsupervised == []:
        raise StandardError('you must provide non empty list for unsupervised')
    execute(args.dataset,
            args.n_hidden_u,
            args.unsupervised,
            int(args.num_epochs),
            args.learning_rate,
            args.save)


if __name__ == '__main__':
    main()
