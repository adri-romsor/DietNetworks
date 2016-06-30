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


def iterate_minibatches(inputs, targets, which_set, batchsize, supervised,
                        unsupervised, split=[0.2, 0.2], shuffle=False):
    assert len(inputs) == len(targets)

    n_total = inputs.shape[0] if supervised else inputs.shape[1]
    n_valid = int(split[0]*n_total)
    n_test = int(split[1]*n_total)

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

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
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


def monitoring(minibatches, dataset_name, error_fn, monitoring_labels):

    monitoring_values = np.zeros(len(monitoring_labels), dtype="float32")
    global_batches = 0

    for batch in minibatches:
        # Update monitored values
        out = error_fn(*batch)
        monitoring_values = monitoring_values + out
        global_batches += 1

    # Print monitored values
    monitoring_values /= global_batches
    for (label, val) in zip(monitoring_labels, monitoring_values):
        print ("  {} {}:\t\t{:.6f}".format(dataset_name, label, val))

    return monitoring_values

# Main program
def execute(training, dataset, n_hidden_u, n_hidden_t, n_hidden_s,
            embedding_source=None, supervised=True,
            num_epochs=500, unsupervised=None):
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

        # ALERT: check that x and y are not transposed!

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

        """
        # Preprocess the targets by removing the mean of the training labels
        mean_y = y_train.mean()
        y_train -= mean_y
        y_valid -= mean_y
        y_test -= mean_y
        print("Labels have been centered by removing %f cm." % mean_y)
        """

        # Make sure all is well
        print(x_sup.shape, x_test.shape, x_unsup.shape)
        print(x_train.shape, x_valid.shape, x_test.shape)

    else:
        print("Unknown dataset")
        return

    n_samples, n_feats = x.shape
    n_classes = y.max() + 1
    batch = 100

    save_path = '/data/lisatmp4/carriepl/FeatureSelection/'

    # Prepare Theano variables for inputs and targets
    input_var_sup = T.matrix('input_sup')
    input_var_unsup = T.matrix('input_unsup')
    target_var_sup = T.ivector('target_sup')
    lr = theano.shared(np.float32(1e-3), 'learning_rate')

    # Build model
    print("Building model")

    assert len(n_hidden_u) > 0
    assert len(n_hidden_t) > 0

    # Define unsupervised network
    if not embedding_source:
        encoder_net = InputLayer((batch, n_samples),
                                 input_var_unsup)
        for out in n_hidden_u:
            encoder_net = DenseLayer(encoder_net, num_units=out)
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

    if supervised:
        # f_theta
        for hid in n_hidden_t:
            encoder_net = DenseLayer(encoder_net, num_units=hid)
        final_feat_emb = lasagne.layers.get_output(encoder_net)

        # Define supervised network
        discrim_net = InputLayer((batch, n_feats), input_var_sup)
        discrim_net = DenseLayer(discrim_net, num_units=n_hidden_t[-1],
                                 W=final_feat_emb)

        for hid in n_hidden_s:
            discrim_net = DenseLayer(discrim_net, num_units=hid)

        discrim_net = DenseLayer(discrim_net, num_units=n_classes,
                                 nonlinearity=softmax)

    # Create a loss expression for training
    print("Building and compiling training functions")

    # Expressions required for training
    loss_sup = 0
    loss_sup_det = 0
    loss_auto = 0
    loss_auto_det = 0
    loss_epls = 0
    loss_epls_det = 0
    params = []
    inputs = []
    if supervised:
        prediction = lasagne.layers.get_output(discrim_net)
        prediction_det = lasagne.layers.get_output(discrim_net,
                                                   deterministic=True)

        # ALERT!
        loss_sup = lasagne.objectives.categorical_crossentropy(
            prediction, target_var_sup).mean()
        loss_sup_det = lasagne.objectives.categorical_crossentropy(
            prediction_det, target_var_sup).mean()

        params += lasagne.layers.get_all_params(discrim_net, trainable=True)

        inputs += [input_var_sup, target_var_sup]
    if "autoencoder" in unsupervised:
        reconstruction = lasagne.layers.get_output(decoder_net)
        reconstruction_det = lasagne.layers.get_output(decoder_net,
                                                       deterministic=True)

        # ALERT!
        loss_auto = lasagne.objectives.squared_error(
            reconstruction,
            input_var_unsup).mean()
        loss_auto_det = lasagne.objectives.squared_error(
            reconstruction_det,
            input_var_unsup).mean()

        params += lasagne.layers.get_all_params(decoder_net, trainable=True)
        inputs += [input_var_unsup]
    if "epls" in unsupervised:
        raise NotImplementedError

    loss = loss_sup + loss_auto + loss_epls
    loss_det = loss_sup_det + loss_auto_det + loss_epls_det

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
    monitor_labels = ['total_loss_deterministic']
    val_outputs = [loss_det]
    if supervised:
        test_class = T.argmax(prediction_det, axis=1)
        test_acc = T.mean(T.eq(test_class, target_var_sup),
                          dtype=theano.config.floatX) * 100.

        val_outputs += [loss_sup_det, test_acc]
        monitor_labels += ["loss_sup_deterministic", "accuracy"]

    if "autoencoder" in unsupervised:
        val_outputs += [loss_auto_det]
        monitor_labels += ["recon. loss"]

    if "epls" in unsupervised:
        raise NotImplementedError

    val_fn = theano.function(inputs,
                             val_outputs,
                             on_unused_input='ignore')

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    best_valid = 0.0
    max_patience = 100
    loss_epoch = 0
    patience = 0

    train_loss = []
    valid_loss = []
    valid_acc = []

    # ALERT
    nb_minibatches = (n_samples if supervised else n_feats)/batch

    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data to updates
        # the parameters:
        start_time = time.time()
        print("Epoch {} of {}".format(epoch + 1, num_epochs))

        for batch in iterate_minibatches(x, y, 'train', batch, supervised,
                                         unsupervised is not None, shuffle=True):
            loss_epoch += train_fn(*batch)

        loss_epoch /= nb_minibatches
        train_loss += loss_epoch

        # Only monitor on the validation set if training in a supervised way
        # otherwise the dimensions will not match.
        valid_minibatches = iterate_minibatches(x, y,
                                                'valid',
                                                batch,
                                                supervised,
                                                unsupervised is not None,
                                                shuffle=False)

        valid_err = monitoring(valid_minibatches, "valid", val_fn,
                               monitor_labels)
        valid_loss += valid_err[0]
        if supervised:
            valid_acc += valid_err[2]

        if valid_err < best_valid:
            best_valid = valid_err
            patience = 0

            # If there are test labels, perform the monitoring. Else, print
            # the test predictions for external evaluation.
            if supervised:
                test_minibatches = iterate_minibatches(x_test, y_test,
                                                       'test',
                                                       batch,
                                                       supervised,
                                                       unsupervised is not None,
                                                       shuffle=False)
                valid_err = monitoring(test_minibatches, "test", val_fn,
                                       monitor_labels)

            # Save network weights to a file
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            if supervised:
                np.savez(save_path+'model_supervised.npz',
                         *lasagne.layers.get_all_param_values(discrim_net))
            if not embedding_source:
                # feature embedding prediction
                for batch in iterate_minibatches(x, y, 'train', n_feats,
                                                 not supervised,
                                                 unsupervised is not None,
                                                 shuffle=False):
                    pred = pred_feat_emb(*batch)
                np.savez(save_path+'feature_emebedding.npz', pred)
        else:
            patience += 1

        if patience == max_patience:
            break

    print("  total time:\t\t\t{:.3f}s".format(time.time() - start_time))


def main():
    parser = argparse.ArgumentParser(description="""Implementation of the
                                     feature selection v2""")
    parser.add_argument('--training',
                        default='supervised',
                        help='Type of training.')
    parser.add_argument('--dataset',
                        default='protein_binding',
                        help='Dataset.')
    parser.add_argument('--n_output',
                        default=100,
                        help='Output dimension.')
    parser.add_argument('--embedding_source',
                        default=None,
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

    unsupervised = ['autoencoder']
    supervised = False
    n_hidden_u = [50, 50]
    n_hidden_t = [50]
    n_hidden_s = []

    execute(args.training, args.dataset, n_hidden_u, n_hidden_t, n_hidden_s,
            args.embedding_source, supervised, int(args.num_epochs), unsupervised=unsupervised)


if __name__ == '__main__':
    main()
