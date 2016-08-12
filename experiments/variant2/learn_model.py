from __future__ import print_function
import argparse
import time
import os

import lasagne
from lasagne.layers import DenseLayer, InputLayer, DropoutLayer, BatchNormLayer
from lasagne.nonlinearities import (sigmoid, softmax, tanh, linear, rectify,
                                    leaky_rectify, very_leaky_rectify)
from lasagne.init import Uniform
import numpy as np
import theano
import theano.tensor as T

from feature_selection.experiments.common import dataset_utils, imdb, \
    dragonn_data


# Mini-batch iterator function
def iterate_minibatches(inputs, targets, batchsize,
                        shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    indices = np.arange(inputs.shape[0])
    if shuffle:
        indices = np.random.permutation(inputs.shape[0])
    for i in range(0, inputs.shape[0]-batchsize+1, batchsize):
        yield inputs[indices[i:i+batchsize], :],\
            targets[indices[i:i+batchsize]]


def iterate_testbatches(inputs, batchsize, shuffle=False):
    indices = np.arange(inputs.shape[0])
    if shuffle:
        indices = np.random.permutation(inputs.shape[0])
    for i in range(0, inputs.shape[0]-batchsize+1, batchsize):
        yield inputs[indices[i:i+batchsize], :]


def get_precision_recall_cutoff(predictions, targets):

    prev_threshold = 0.00
    threshold_inc = 0.10

    while True:
        if prev_threshold > 1.000:
            cutoff = 0.0
            break

        threshold = prev_threshold + threshold_inc
        tp = ((predictions >= threshold) * (targets == 1)).sum()
        fp = ((predictions >= threshold) * (targets == 0)).sum()
        fn = ((predictions < threshold) * (targets == 1)).sum()

        precision = float(tp) / (tp + fp + 1e-20)
        recall = float(tp) / (tp + fn + 1e-20)

        if precision > recall:
            if threshold_inc < 0.001:
                cutoff = recall
                break
            else:
                threshold_inc /= 10
        else:
            prev_threshold += threshold_inc

    return cutoff


# Monitoring function
def monitoring(minibatches, which_set, error_fn, monitoring_labels, PR=True):
    print('-'*20 + which_set + ' monit.' + '-'*20)
    monitoring_values = np.zeros(len(monitoring_labels), dtype="float32")
    global_batches = 0

    targets = []
    predictions = []

    for batch in minibatches:
        # Update monitored values
        out = error_fn(*batch)
        if PR:
            monitoring_values = monitoring_values + out[1:]
            predictions.append(out[0])
            targets.append(batch[1])
        else:
            monitoring_values = monitoring_values + out
        global_batches += 1

    if PR:
        # Compute the precision-recall breakoff point
        predictions = np.vstack(predictions)
        targets = np.vstack(targets)
        cutoff = get_precision_recall_cutoff(predictions, targets)

    # Print monitored values
    monitoring_values /= global_batches
    for (label, val) in zip(monitoring_labels, monitoring_values):
        print ("  {} {}:\t\t{:.6f}".format(which_set, label, val))
    if PR:
        print ("  {} precis/recall cutoff:\t{:.6f}".format(which_set, cutoff))

    return monitoring_values


# Main program
def execute(dataset, n_hidden_u, n_hidden_t_enc, n_hidden_t_dec, n_hidden_s,
            embedding_source=None,
            num_epochs=500, learning_rate=.001, learning_rate_annealing=1.0,
            gamma=1, save_path='/Tmp/romerosa/feature_selection/newmodel/'):

    # Load the dataset
    print("Loading data")
    splits = [0.6, 0.2]  # This will split the data into [60%, 20%, 20%]
    if dataset == 'protein_binding':
        data = dataset_utils.load_protein_binding(transpose=False,
                                                  splits=splits)
    elif dataset == 'dorothea':
        data = dataset_utils.load_dorothea(transpose=False, splits=None)
    elif dataset == 'opensnp':
        data = dataset_utils.load_opensnp(transpose=False, splits=splits)
    elif dataset == 'reuters':
        data = dataset_utils.load_reuters(transpose=False, splits=splits)
    elif dataset == 'imdb':
        # use feat_type='tfidf' to load tfidf features
        data = imdb.read_from_hdf5(unsupervised=False, feat_type='tfidf')
    elif dataset == 'dragonn':
        data = dragonn_data.load_data(500, 100, 100)
    else:
        print("Unknown dataset")
        return

    if dataset == 'imdb':
        x_train = data.root.train_features
        y_train = data.root.train_labels[:][:, None]
        x_valid = data.root.val_features
        y_valid = data.root.val_labels[:][:, None]
        x_test = data.root.test_features
        y_test = None
        x_nolabel = None
    else:
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test),\
            x_nolabel = data

    if not embedding_source:
        if x_nolabel is None:
            x_unsup = x_train[:5000].transpose()
        else:
            x_unsup = np.vstack((x_train, x_nolabel)).transpose()
        n_samples_unsup = x_unsup.shape[1]
    else:
        x_unsup = None

    # Extract required information from data
    n_samples, n_feats = x_train.shape
    print("Number of features : ", n_feats)
    print("Glorot init : ", 2.0 / n_feats)
    n_targets = y_train.shape[1]

    # Set some variables
    batch_size = 128

    # Preparing folder to save stuff
    save_path = save_path + dataset + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Prepare Theano variables for inputs and targets
    input_var_sup = T.matrix('input_sup')
    input_var_unsup = theano.shared(x_unsup, 'input_unsup')  # x_unsup TBD
    target_var_sup = T.imatrix('target_sup')
    lr = theano.shared(np.float32(learning_rate), 'learning_rate')

    # Build model
    print("Building model")

    # Some checkings
    assert len(n_hidden_u) > 0
    assert len(n_hidden_t_enc) > 0
    assert len(n_hidden_t_dec) > 0
    assert len(n_hidden_u) > 0
    assert n_hidden_t_dec[-1] == n_hidden_t_enc[-1]

    # Build unsupervised network
    if not embedding_source:
        encoder_net = InputLayer((n_feats, n_samples_unsup), input_var_unsup)
        for i, out in enumerate(n_hidden_u):
            encoder_net = DenseLayer(encoder_net, num_units=out,
                                     nonlinearity=rectify)
        feat_emb = lasagne.layers.get_output(encoder_net)
        pred_feat_emb = theano.function([], feat_emb)
        alpha_emb = 0.0095  # may need to adapt this constant
    else:
        feat_emb_val = np.load(save_path + embedding_source).items()[0][1]
        # feat_emb_val = np.random.randn(123333, 100).astype('float32')
        feat_emb = theano.shared(feat_emb_val, 'feat_emb')
        encoder_net = InputLayer((n_feats, n_hidden_u[-1]), feat_emb)
        alpha_emb = 0.0015  # may need to adapt this constant
    # Build transformations (f_theta, f_theta') network and supervised network
    # f_theta (ou W_enc)
    encoder_net_W_enc = encoder_net
    for hid in n_hidden_t_enc:
        encoder_net_W_enc = DenseLayer(encoder_net_W_enc, num_units=hid,
                                       nonlinearity=tanh, W=Uniform(alpha_emb))
    # layers_net_W_enc = lasagne.layers.get_all_layers(encoder_net_W_enc)
    # activs_net_W_enc = lasagne.layers.get_output(layers_net_W_enc)
    enc_feat_emb = lasagne.layers.get_output(encoder_net_W_enc)
    # enc_feat_emb = activs_net_W_enc[-1]

    # f_theta' (ou W_dec)
    encoder_net_W_dec = encoder_net
    for hid in n_hidden_t_dec:
        encoder_net_W_dec = DenseLayer(encoder_net_W_dec, num_units=hid,
                                       nonlinearity=tanh, W=Uniform(alpha_emb))
    # layers_net_W_dec = lasagne.layers.get_all_layers(encoder_net_W_dec)
    # activs_net_W_dec = lasagne.layers.get_output(layers_net_W_dec)
    dec_feat_emb = lasagne.layers.get_output(encoder_net_W_dec)
    # dec_feat_emb = activs_net_W_dec[-1]

    # Supervised network
    discrim_net = InputLayer((batch_size, n_feats), input_var_sup)
    # discrim_net = BatchNormLayer(discrim_net, gamma=lasagne.init.Constant(.1))
    discrim_net = DenseLayer(discrim_net, num_units=n_hidden_t_enc[-1],
                             W=enc_feat_emb, nonlinearity=rectify)

    # reconstruct the input using dec_feat_emb
    reconst_net = DenseLayer(discrim_net, num_units=n_feats,
                             W=dec_feat_emb.T)

    # predicting labels
    for hid in n_hidden_s:
        # discrim_net = BatchNormLayer(discrim_net)
        discrim_net = DenseLayer(discrim_net, num_units=hid)
    discrim_net = DenseLayer(discrim_net, num_units=n_targets,
                             nonlinearity=sigmoid)

    print("Building and compiling training functions")
    # Some variables
    loss_sup = 0
    loss_sup_det = 0

    # Build and compile training functions

    # network activations
    # net_layers = lasagne.layers.get_all_layers([discrim_net, reconst_net])
    # net_activs = lasagne.layers.get_output(net_layers)
    # prediction = net_activs[len(lasagne.layers.get_all_layers(discrim_net))-1]
    # reconstruction = net_activs[-1]
    # net_activs += activs_net_W_enc + activs_net_W_dec[1:]

    prediction = lasagne.layers.get_output(discrim_net)
    prediction_det = lasagne.layers.get_output(discrim_net,
                                               deterministic=True)

    # Supervised loss
    loss_sup = lasagne.objectives.binary_crossentropy(
        prediction, target_var_sup).mean()
    loss_sup_det = lasagne.objectives.binary_crossentropy(
        prediction_det, target_var_sup).mean()

    inputs = [input_var_sup, target_var_sup]

    # Unsupervised reconstruction loss
    reconstruction = lasagne.layers.get_output(reconst_net)
    reconstruction_det = lasagne.layers.get_output(reconst_net,
                                                   deterministic=True)
    reconst_loss = lasagne.objectives.squared_error(
        reconstruction,
        input_var_sup).mean()
    reconst_loss_det = lasagne.objectives.squared_error(
        reconstruction_det,
        input_var_sup).mean()

    params = lasagne.layers.get_all_params([discrim_net, reconst_net,
                                            encoder_net_W_dec,
                                            encoder_net_W_enc],
                                           trainable=True)

    # Combine losses
    loss = loss_sup + gamma*reconst_loss
    loss_det = loss_sup_det + gamma*reconst_loss_det

    # Compute network updates
    updates = lasagne.updates.rmsprop(loss,
                                      params,
                                      learning_rate=lr)
    # updates = lasagne.updates.sgd(loss,
    #                               params,
    #                               learning_rate=lr)
    # updates = lasagne.updates.momentum(loss, params,
    #                                    learning_rate=lr, momentum=0.0)

    # Compile training function
    train_fn = theano.function(inputs, loss, updates=updates,
                               on_unused_input='ignore')

    # Monitoring gradients and activations
    # acts = theano.function(inputs, net_activs, on_unused_input='ignore')
    # layers_grads = T.grad(loss, net_activs)
    # layers_grads_norm = [gd.norm(2) for gd in layers_grads]
    # net_grads = theano.function(inputs, layers_grads)
    # net_grads_norm = theano.function(inputs, layers_grads_norm)

    # Supervised functions
    test_pred = T.gt(prediction_det, 0.5)
    predict = theano.function([input_var_sup], test_pred)
    test_acc = T.mean(T.eq(test_pred, target_var_sup),
                      dtype=theano.config.floatX) * 100.

    # Expressions required for test
    monitor_labels = ["total_loss_det", "loss_sup_det", "accuracy",
                      "recon. loss", "enc_w_mean", "enc_w_var"]
    val_outputs = [loss_det, loss_sup_det, test_acc, reconst_loss_det,
                   enc_feat_emb.mean(), enc_feat_emb.var()]

    # Compile validation function
    val_fn = theano.function(inputs,
                             [prediction_det] + val_outputs,
                             on_unused_input='ignore')

    # Finally, launch the training loop.
    print("Starting training...")

    # Some variables
    max_patience = 100
    patience = 0

    train_loss = []
    valid_loss = []
    valid_loss_sup = []
    valid_reconst_loss = []
    valid_acc = []

    start_training = time.time()
    # grads_norms = np.zeros((1, len(net_activs)))
    for epoch in range(num_epochs):
        start_time = time.time()
        print("Epoch {} of {}".format(epoch+1, num_epochs))
        nb_minibatches = 0
        loss_epoch = 0
        # grads_norms = np.zeros((1, len(net_layers)))
        # Train pass
        for batch in iterate_minibatches(x_train, y_train,
                                         batch_size,
                                         shuffle=True):
            loss_epoch += train_fn(*batch)
            # grads_norms = np.vstack((grads_norms,
            #                          np.array(net_grads_norm(*batch))))
            nb_minibatches += 1
        # print ("  Train loss: \t\t\t{:.6f}".format(loss_epoch / nb_minibatches))
        loss_epoch /= nb_minibatches
        train_loss += [loss_epoch]

        # Monitoring on the training set
        train_minibatches = iterate_minibatches(x_train, y_train,
                                                batch_size,
                                                shuffle=False)
        monitoring(train_minibatches, "train", val_fn,
                   monitor_labels)

        # Monitoring on the validation set
        valid_minibatches = iterate_minibatches(x_valid, y_valid,
                                                batch_size,
                                                shuffle=False)

        valid_err = monitoring(valid_minibatches, "valid", val_fn,
                               monitor_labels)
        valid_loss += [valid_err[0]]
        valid_loss_sup += [valid_err[1]]
        valid_acc += [valid_err[2]]
        valid_reconst_loss += [valid_err[3]]

        # Early stopping
        if epoch == 0:
            best_valid = valid_loss[epoch]
        elif valid_loss[epoch] < best_valid:
            best_valid = valid_loss[epoch]
            patience = 0

            # Save stuff
            np.savez(save_path+'model_feat_sel.npz',
                     *lasagne.layers.get_all_param_values([reconst_net,
                                                           discrim_net]))
            np.savez(save_path + "errors_supervised.npz",
                     train_loss, valid_loss, valid_loss_sup, valid_acc,
                     valid_reconst_loss)
        else:
            patience += 1

        # End training
        if patience == max_patience or epoch == num_epochs-1:
            print("Ending training")
            # Load best model
            if not os.path.exists(save_path + 'model_feat_sel.npz'):
                print("No saved model to be tested and/or generate"
                      " the embedding !")
            else:
                with np.load(save_path + 'model_feat_sel.npz',) as f:
                    param_values = [f['arr_%d' % i]
                                    for i in range(len(f.files))]
                    nlayers = len(lasagne.layers.get_all_params([reconst_net,
                                                                discrim_net]))
                    lasagne.layers.set_all_param_values([reconst_net,
                                                        discrim_net],
                                                        param_values[:nlayers])
            if not embedding_source:
                # Save embedding
                pred = pred_feat_emb()
                np.savez(save_path+'feature_embedding.npz', pred)

            # Test
            if y_test is not None:
                test_minibatches = iterate_minibatches(x_test, y_test,
                                                       batch_size,
                                                       shuffle=False)

                test_err = monitoring(test_minibatches, "test", val_fn,
                                      monitor_labels)
            else:
                for minibatch in iterate_testbatches(x_test,
                                                     batch_size,
                                                     shuffle=False):
                    test_predictions = []
                    test_predictions += [predict(minibatch)]
                np.savez(save_path+'test_predictions.npz', test_predictions)

            # Stop
            print("  epoch time:\t\t\t{:.3f}s \n".format(time.time() -
                                                         start_time))
            break

        print("  epoch time:\t\t\t{:.3f}s \n".format(time.time() - start_time))

        # Anneal the learning rate
        lr.set_value(float(lr.get_value() * learning_rate_annealing))

    # Print all final errors for train, validation and test
    print("Training time:\t\t\t{:.3f}s".format(time.time() - start_training))


def main():
    parser = argparse.ArgumentParser(description="""Implementation of the
                                     feature selection v2""")
    parser.add_argument('--dataset',
                        default='imdb',
                        help='Dataset.')
    parser.add_argument('--n_hidden_u',
                        default=[100],
                        help='List of unsupervised hidden units.')
    parser.add_argument('--n_hidden_t_enc',
                        default=[100],
                        help='List of theta transformation hidden units.')
    parser.add_argument('--n_hidden_t_dec',
                        default=[100],
                        help='List of theta_prime transformation hidden units')
    parser.add_argument('--n_hidden_s',
                        default=[100],
                        help='List of supervised hidden units.')
    parser.add_argument('--embedding_source',
                        default=None,
                        help='Source for the feature embedding. Either' +
                             'None or the name of a file from which' +
                             'to load a learned embedding')
    parser.add_argument('--num_epochs',
                        '-ne',
                        type=int,
                        default=100,
                        help="""Int to indicate the max'
                        'number of epochs.""")
    parser.add_argument('--learning_rate',
                        '-lr',
                        type=float,
                        default=.00001,
                        help="""Float to indicate learning rate.""")
    parser.add_argument('--learning_rate_annealing',
                        '-lra',
                        type=float,
                        default=1.0,
                        help="Float to indicate learning rate annealing rate.")
    parser.add_argument('--gamma',
                        '-g',
                        type=float,
                        default=0,
                        help="""reconst_loss coeff.""")
    parser.add_argument('--save',
                        default='/Tmp/carriepl/feature_selection/v4/',
                        help='Path to save results.')

    args = parser.parse_args()

    execute(args.dataset,
            args.n_hidden_u,
            args.n_hidden_t_enc,
            args.n_hidden_t_dec,
            args.n_hidden_s,
            args.embedding_source,
            int(args.num_epochs),
            args.learning_rate,
            args.learning_rate_annealing,
            args.gamma,
            args.save)


if __name__ == '__main__':
    main()
