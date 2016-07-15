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
def iterate_minibatches(inputs, targets, batchsize,
                        shuffle=False):
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


# Monitoring function
def monitoring(minibatches, which_set, error_fn, monitoring_labels):

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
        print ("  {} {}:\t\t{:.6f}".format(which_set, label, val))

    return monitoring_values


# Main program
def execute(dataset, n_hidden_u, n_hidden_t_enc, n_hidden_t_dec, n_hidden_s,
            embedding_source=None,
            num_epochs=500, learning_rate=.001, gamma=1,
            save_path='/Tmp/romerosa/feature_selection/newmodel/'):

    ######### Put your data loading script in this block ############


    #Define x_train, y_train, x_val, y_val, x_test, y_test, x_unsup #


    #################################################################

    # Extract required information from data
    n_samples, n_feats = x.shape  # x and y TBD
    n_classes = y.max() + 1

    # Set some variables
    batch_size = 128

    # Preparing folder to save stuff
    save_path = save_path + dataset + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Prepare Theano variables for inputs and targets
    input_var_sup = T.matrix('input_sup')
    input_var_unsup = theano.shared(x_unsup, 'input_unsup')  # x_unsup TBD
    target_var_sup = T.ivector('target_sup')
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
        encoder_net = InputLayer((n_feats, n_samples),
                                 input_var_unsup.get_value())
        for out in n_hidden_u:
            encoder_net = DenseLayer(encoder_net, num_units=out,
                                     nonlinearity=sigmoid)
        feat_emb = lasagne.layers.get_output(encoder_net)
        pred_feat_emb = theano.function([], feat_emb)

    else:
        feat_emb_val = np.load(save_path + embedding_source).items()[0][1]
        feat_emb = theano.shared(feat_emb_val, 'feat_emb')
        encoder_net = InputLayer((n_feats, n_hidden_u[-1]),
                                 feat_emb.get_value())

    # Build transformations (f_theta, f_theta') network and supervised network
    # f_theta (ou W_enc)
    for hid in n_hidden_t_enc:
        encoder_net_W_enc = DenseLayer(encoder_net, num_units=hid)
    enc_feat_emb = lasagne.layers.get_output(encoder_net_W_enc)

    # f_theta' (ou W_dec)
    for hid in n_hidden_t_dec:
        encoder_net_W_dec = DenseLayer(encoder_net, num_units=hid)
    dec_feat_emb = lasagne.layers.get_output(encoder_net_W_dec)

    # Supervised network
    discrim_net = InputLayer((batch_size, n_feats), input_var_sup)
    discrim_net = DenseLayer(discrim_net, num_units=n_hidden_t_enc[-1],
                             W=enc_feat_emb)

    # reconstruct the input using dec_feat_emb
    reconst_net = DenseLayer(discrim_net, num_units=n_feats,
                             W=dec_feat_emb.T)

    # predicting labels
    for hid in n_hidden_s:
        discrim_net = DenseLayer(discrim_net, num_units=hid)
    discrim_net = DenseLayer(discrim_net, num_units=n_classes,
                             nonlinearity=softmax)

    print("Building and compiling training functions")
    # Some variables
    loss_sup = 0
    loss_sup_det = 0
    # Build and compile training functions

    # Supervised loss
    prediction = lasagne.layers.get_output(discrim_net)
    prediction_det = lasagne.layers.get_output(discrim_net,
                                               deterministic=True)
    loss_sup = lasagne.objectives.categorical_crossentropy(
        prediction, target_var_sup).mean()
    loss_sup_det = lasagne.objectives.categorical_crossentropy(
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

    params = lasagne.layers.get_all_params([discrim_net, reconst_net],
                                           trainable=True)

    # Combine losses
    loss = loss_sup + gamma*reconst_loss
    loss_det = loss_sup_det + reconst_loss_det
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

    # Supervised functions
    test_class = T.argmax(prediction_det, axis=1)
    test_acc = T.mean(T.eq(test_class, target_var_sup),
                      dtype=theano.config.floatX) * 100.

    # Expressions required for test
    monitor_labels = ["total_loss_det", "loss_sup_det", "accuracy",
                      "recon. loss"]
    val_outputs = [loss_det, loss_sup_det, test_acc, reconst_loss_det]

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
    valid_loss_sup = []
    valid_reconst_loss = []
    valid_acc = []

    nb_minibatches = n_samples
    start_training = time.time()
    for epoch in range(num_epochs):
        start_time = time.time()
        print("Epoch {} of {}".format(epoch+1, num_epochs))

        loss_epoch = 0

        # Train pass
        for batch in iterate_minibatches(x_train, y_train,  # TBD
                                         batch_size,
                                         shuffle=True):
            loss_epoch += train_fn(*batch)

        loss_epoch /= nb_minibatches
        train_loss += [loss_epoch]

        # Validation pass
        valid_minibatches = iterate_minibatches(x_val, y_val,  # TBD
                                                batch_size,
                                                shuffle=False)

        valid_err = monitoring(valid_minibatches, "valid", val_fn,
                               monitor_labels)

        valid_loss += [valid_err[0]]
        valid_loss_sup += [valid_err[1]]
        valid_acc += [valid_err[2]]
        valid_reconst_loss += [valid_err[3]]

        # Eearly stopping
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
            print("   Ending training")
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
            test_minibatches = iterate_minibatches(x_test, y_test,
                                                   batch_size,
                                                   shuffle=False)

            test_err = monitoring(test_minibatches, "test", val_fn,
                                  monitor_labels)
            # Stop
            print("  epoch time:\t\t\t{:.3f}s".format(time.time() -
                                                      start_time))
            break

        print("  epoch time:\t\t\t{:.3f}s".format(time.time() - start_time))

    # Print all final errors for train, validation and test
    print("Training time:\t\t\t{:.3f}s".format(time.time() - start_training))


def main():
    parser = argparse.ArgumentParser(description="""Implementation of the
                                     feature selection v2""")
    parser.add_argument('--dataset',
                        default='imdb',
                        help='Dataset.')
    parser.add_argument('--n_hidden_u',
                        default=[40],
                        help='List of unsupervised hidden units.')
    parser.add_argument('--n_hidden_t_enc',
                        default=[100],
                        help='List of theta transformation hidden units.')
    parser.add_argument('--n_hidden_t_dec',
                        default=[100],
                        help='List of theta_prime transformation hidden units')
    parser.add_argument('--n_hidden_s',
                        default=[60],
                        help='List of supervised hidden units.')
    parser.add_argument('--embedding_source',
                        default=None,
                        help='Source for the feature embedding. Either' +
                             'None or the name of a file from which' +
                             'to load a learned embedding')
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
    parser.add_argument('--gamma',
                        '-lr',
                        type=float,
                        default=1,
                        help="""reconst_loss coeff.""")
    parser.add_argument('--save',
                        default='/Tmp/erraqaba/feature_selection/v4/',
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
            args.gamma,
            args.save)


if __name__ == '__main__':
    main()
