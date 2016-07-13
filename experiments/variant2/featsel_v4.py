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


# Mini-batch iterator function
def iterate_minibatches(inputs, targets, which_set, batchsize, supervised,
                        unsupervised, split=[0.2, 0.2], shuffle=False):
    assert len(inputs) == len(targets)

    n_total = inputs.shape[0] if supervised else inputs.shape[1]

    if supervised:
        n_valid = int(split[0]*n_total)
        n_test = int(split[1]*n_total)
    else:
        n_valid = int(split[0]*n_total)
        n_test = 0  # we don't need test for the unsupervised-only case

    if which_set == 'train':
        set_indices = slice(n_valid + n_test, n_total)
    elif which_set == 'valid':
        set_indices = slice(0, n_valid)
    elif which_set == 'test':
        set_indices = slice(n_valid, n_valid + n_test)

    if supervised and unsupervised:
        inputs_iterable = inputs[set_indices]
        targets = targets[set_indices]
        inputs_unsup = inputs.transpose()
    elif unsupervised:
        inputs_iterable = inputs.transpose()[set_indices]
    elif supervised:
        inputs_iterable = inputs[set_indices]
        targets = targets[set_indices]

    if shuffle:
        indices = np.arange(len(inputs_iterable))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs_iterable) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        if supervised and unsupervised:
            yield inputs_iterable[excerpt], targets[excerpt], inputs_unsup
        elif unsupervised:
            yield inputs_iterable[excerpt]
        elif supervised:
            yield inputs_iterable[excerpt], targets[excerpt]


# Monitoring function
def monitoring(minibatches, which_set, error_fn, monitoring_labels,
               supervised):

    monitoring_values = np.zeros(len(monitoring_labels), dtype="float32")
    global_batches = 0

    for batch in minibatches:
        # Update monitored values
        if supervised:
            out = error_fn(*batch)
        else:
            out = error_fn(batch)

        monitoring_values = monitoring_values + out
        global_batches += 1

    # Print monitored values
    monitoring_values /= global_batches
    for (label, val) in zip(monitoring_labels, monitoring_values):
        print ("  {} {}:\t\t{:.6f}".format(which_set, label, val))

    return monitoring_values


# Main program
def execute(dataset, n_hidden_u, n_hidden_t, n_hidden_s,
            embedding_source=None, supervised=True,
            unsupervised=[], num_epochs=500, learning_rate=.001,
            save_path='/Tmp/romerosa/feature_selection/newmodel/'):

    # Load the dataset
    print("Loading data")
    if dataset == 'protein_binding':
        from experiments.common.protein_loader import load_data
        x, y = load_data()
    elif dataset == 'dorothea':
        from feature_selection.experiments.common.dorothea import load_data
        x_train, y_train = load_data('train', 'standard', False, 'numpy')
        x_valid, y_valid = load_data('valid', 'standard', False, 'numpy')

        # WARNING : The dorothea dataset has no test labels
        x_test = load_data('test', 'standard', False, 'numpy')
        y_test = None

    elif dataset == 'opensnp':
        from feature_selection import aggregate_dataset

        # This splits the data into [0.6, 0.2, 0.2] for the supervised examples
        # and puts half of the unsupervised examples in the training set
        # (because otherwise it would be way to expensive memory-wise)
        data = aggregate_dataset.load_data23andme_baselines(split=0.8)
        (x_sup, x_sup_labels), (x_test, y_test), x_unsup = data

        nb_sup_train = int(x_sup.shape[0] * 0.75)
        nb_unsup_train = int(x_unsup.shape[0] * 0.5)

        x_train = np.vstack((x_sup[:nb_sup_train], x_unsup[:nb_unsup_train]))
        x_valid = x_sup[nb_sup_train:]

        y_train = np.hstack((x_sup_labels[:nb_sup_train],
                             -np.ones((nb_unsup_train,))))
        y_valid = x_sup_labels[nb_sup_train:]

        x_train = x_sup[:nb_sup_train]
        y_train = x_sup_labels[:nb_sup_train]

        # Shuffle x_train and y_train together
        np.random.seed(0)
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)

        x_train = x_train[indices]
        y_train = y_train[indices]

        # Standardize the dtype
        x_train = x_train.astype("float32")
        x_valid = x_valid.astype("float32")
        x_test = x_test.astype("float32")
        y_train = y_train.astype("float32")
        y_valid = y_valid.astype("float32")
        y_test = y_test.astype("float32")

        # Make sure all is well
        print(x_sup.shape, x_test.shape, x_unsup.shape)
        print(x_train.shape, x_valid.shape, x_test.shape)

    else:
        print("Unknown dataset")
        return

    # Ensure that the targets are a matrix and not a vector
    if 'y' in locals():
        if y.ndim == 1:
            y = y[:, None]
    if 'y_train' in locals():
        if y_train.ndim == 1:
            y_train = y_train[:, None]
    if 'y_valid' in locals():
        if y_valid.ndim == 1:
            y_valid = y_valid[:, None]
    if 'y_test' in locals():
        if y_test.ndim == 1:
            y_test = y_test[:, None]

    # Extract required information from data
    n_samples, n_feats = x.shape
    n_binary_classifications = y.shape[1]

    # Set some variables
    batch_size = 100

    # Preparing folder to save stuff
    save_path = save_path + dataset + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Prepare Theano variables for inputs and targets
    input_var_sup = T.matrix('input_sup')
    input_var_unsup = T.matrix('input_unsup')
    target_var_sup = T.ivector('target_sup')
    lr = theano.shared(np.float32(learning_rate), 'learning_rate')

    # Build model
    print("Building model")

    # Some checkings
    assert len(n_hidden_u) > 0
    assert len(n_hidden_t) > 0
    if embedding_source is not None and unsupervised != []:
        raise ValueError('If we have an embedding_source, train' +
                         'supervised only!')

    # Build unsupervised network
    if not embedding_source:
        encoder_net = InputLayer((batch_size if not supervised else n_feats,
                                  n_samples),
                                 input_var_unsup)
        for out in n_hidden_u:
            encoder_net = DenseLayer(encoder_net, num_units=out,
                                     nonlinearity=sigmoid)
        feat_emb = lasagne.layers.get_output(encoder_net)
        pred_feat_emb = theano.function([input_var_unsup], feat_emb)

        if 'autoencoder' in unsupervised:
            decoder_net = encoder_net
            for i in range(len(n_hidden_u)-2, -1, -1):
                decoder_net = DenseLayer(decoder_net, num_units=n_hidden_u[i],
                                         nonlinearity=sigmoid)
            decoder_net = DenseLayer(decoder_net, num_units=n_samples,
                                     nonlinearity=sigmoid)
            reconstruction = lasagne.layers.get_output(decoder_net)
        if 'epls' in unsupervised:
            raise NotImplementedError

    else:
        feat_emb_val = np.load(save_path + embedding_source).items()[0][1]
        feat_emb = theano.shared(feat_emb_val, 'feat_emb')
        encoder_net = InputLayer((n_feats, n_hidden_u[-1]),
                                 feat_emb.get_value())

    # Build transformation (f_theta) network and supervised network
    if supervised:
        # f_theta
        for hid in n_hidden_t:
            encoder_net = DenseLayer(encoder_net, num_units=hid)
        final_feat_emb = lasagne.layers.get_output(encoder_net)

        # Supervised network
        discrim_net = InputLayer((batch_size, n_feats), input_var_sup)
        discrim_net = DenseLayer(discrim_net, num_units=n_hidden_t[-1],
                                 W=final_feat_emb)

        for hid in n_hidden_s:
            discrim_net = DenseLayer(discrim_net, num_units=hid)

        discrim_net = DenseLayer(discrim_net,
                                 num_units=n_binary_classifications,
                                 nonlinearity=sigmoid)

    print("Building and compiling training functions")
    # Some variables
    loss_sup = 0
    loss_sup_det = 0
    loss_auto = 0
    loss_auto_det = 0
    loss_epls = 0
    loss_epls_det = 0
    params = []
    inputs = []

    # Build and compile training functions
    if supervised:
        # Supervised functions
        prediction = lasagne.layers.get_output(discrim_net)
        prediction_det = lasagne.layers.get_output(discrim_net,
                                                   deterministic=True)

        loss_sup = lasagne.objectives.binary_crossentropy(
            prediction, target_var_sup).mean()
        loss_sup_det = lasagne.objectives.binary_crossentropy(
            prediction_det, target_var_sup).mean()

        params += lasagne.layers.get_all_params(discrim_net, trainable=True)

        inputs += [input_var_sup, target_var_sup]
        inputs += [input_var_unsup] if embedding_source is None else []

    if "autoencoder" in unsupervised:
        # Unsupervised reconstruction functions
        reconstruction = lasagne.layers.get_output(decoder_net)
        reconstruction_det = lasagne.layers.get_output(decoder_net,
                                                       deterministic=True)

        loss_auto = lasagne.objectives.squared_error(
            reconstruction,
            input_var_unsup).mean()
        loss_auto_det = lasagne.objectives.squared_error(
            reconstruction_det,
            input_var_unsup).mean()

        params += lasagne.layers.get_all_params(decoder_net, trainable=True)
        inputs += [input_var_unsup] if input_var_unsup not in inputs else []
    if "epls" in unsupervised:
        # Unsupervised epls functions
        raise NotImplementedError

    # Combine losses
    loss = loss_sup + loss_auto + loss_epls
    loss_det = loss_sup_det + loss_auto_det + loss_epls_det

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
    train_fn = theano.function(inputs, loss,
                               updates=updates,
                               on_unused_input='ignore')

    # Expressions required for test
    monitor_labels = ['total_loss_det']
    val_outputs = [loss_det]
    if supervised:
        # Supervised functions
        test_class = T.argmax(prediction_det, axis=1)
        test_acc = T.mean(T.eq(test_class, target_var_sup),
                          dtype=theano.config.floatX) * 100.

        val_outputs += [loss_sup_det, test_acc]
        monitor_labels += ["loss_sup_det", "accuracy"]

    if "autoencoder" in unsupervised:
        # Unsupervised reconstruction functions
        val_outputs += [loss_auto_det]
        monitor_labels += ["recon. loss"]

    if "epls" in unsupervised:
        # Unsupervised epls functions
        raise NotImplementedError

    # Compile validation function
    val_fn = theano.function(inputs,
                             val_outputs,
                             on_unused_input='ignore')

    # Finally, launch the training loop.
    print("Starting training...")

    # Some variables
    max_patience = 100
    patience = 0

    train_loss = []
    valid_loss = []
    valid_loss_auto = []
    valid_loss_epls = []
    valid_acc = []

    nb_minibatches = (n_samples if supervised else n_feats)/batch_size
    start_training = time.time()
    for epoch in range(num_epochs):
        start_time = time.time()
        print("Epoch {} of {}".format(epoch+1, num_epochs))

        loss_epoch = 0

        # Train pass
        for batch in iterate_minibatches(x, y, 'train',
                                         batch_size, supervised,
                                         embedding_source is None,
                                         shuffle=True):
            if supervised:
                loss_epoch += train_fn(*batch)
            else:
                loss_epoch += train_fn(batch)

        loss_epoch /= nb_minibatches
        train_loss += [loss_epoch]

        # Validation pass
        valid_minibatches = iterate_minibatches(x, y, 'valid',
                                                batch_size, supervised,
                                                embedding_source is None,
                                                shuffle=False)

        valid_err = monitoring(valid_minibatches, "valid", val_fn,
                               monitor_labels, supervised)

        valid_loss += [valid_err[0]]
        pos = 1
        if supervised:
            valid_acc += [valid_err[pos]]
            pos += 1
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
            if supervised:
                np.savez(save_path+'model_supervised.npz',
                         *lasagne.layers.get_all_param_values(discrim_net))
                np.savez(save_path + "errors_supervised.npz",
                         train_loss, valid_loss, valid_acc)
            if not embedding_source:
                np.savez(save_path+'model_unsupervised.npz',
                         *lasagne.layers.get_all_param_values(encoder_net))
                np.savez(save_path + "errors_unsupervised.npz",
                         valid_loss_auto, valid_loss_epls)
        else:
            patience += 1

        # End training
        if patience == max_patience or epoch == num_epochs-1:
            print("   Ending training")
            if not embedding_source:
                # Load unsupervised best model
                with np.load(save_path + 'model_unsupervised.npz',) as f:
                    param_values = [f['arr_%d' % i]
                                    for i in range(len(f.files))]
                    nlayers = len(lasagne.layers.get_all_params(encoder_net))
                    lasagne.layers.set_all_param_values(encoder_net,
                                                        param_values[:nlayers])

                # Save embedding
                for batch in iterate_minibatches(x, y, 'train', n_feats,
                                                 False,
                                                 embedding_source is None,
                                                 shuffle=False,
                                                 split=[0., 0.]):
                    pred = pred_feat_emb(batch)
                np.savez(save_path+'feature_embedding.npz', pred)

            # Test model
            if supervised:
                # Load supervised best model
                with np.load(save_path + 'model_supervised.npz',) as f:
                    param_values = [f['arr_%d' % i]
                                    for i in range(len(f.files))]
                    nlayers = len(lasagne.layers.get_all_params(discrim_net))
                    lasagne.layers.set_all_param_values(discrim_net,
                                                        param_values[:nlayers])
                # Test
                test_minibatches = iterate_minibatches(
                    x, y, 'test',
                    batch_size, supervised,
                    embedding_source is None,
                    shuffle=False)
                test_err = monitoring(test_minibatches, "test", val_fn,
                                      monitor_labels, supervised)
            # Stop
            print("  epoch time:\t\t\t{:.3f}s".format(time.time() - start_time))
            break

        print("  epoch time:\t\t\t{:.3f}s".format(time.time() - start_time))

    # Print all final errors for train, validation and test
    print("Training time:\t\t\t{:.3f}s".format(time.time() - start_training))


def main():
    parser = argparse.ArgumentParser(description="""Implementation of the
                                     feature selection v2""")
    parser.add_argument('--dataset',
                        default='protein_binding',
                        help='Dataset.')
    parser.add_argument('--n_hidden_u',
                        default=[100],
                        help='List of unsupervised hidden units.')
    parser.add_argument('--n_hidden_t',
                        default=[100],
                        help='List of theta transformation hidden units.')
    parser.add_argument('--n_hidden_s',
                        default=[100],
                        help='List of supervised hidden units.')
    parser.add_argument('--embedding_source',
                        default=None,
                        help='Source for the feature embedding. Either' +
                             'None or the name of a file from which' +
                             'to load a learned embedding')
    parser.add_argument('--supervised',
                        default=True,
                        help='Add supervised network and train it.')
    parser.add_argument('--unsupervised',
                        default=[],
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

    execute(args.dataset,
            args.n_hidden_u,
            args.n_hidden_t,
            args.n_hidden_s,
            args.embedding_source,
            args.supervised,
            args.unsupervised,
            int(args.num_epochs),
            args.learning_rate,
            args.save)


if __name__ == '__main__':
    main()
