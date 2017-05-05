#!/usr/bin/env python
from __future__ import print_function
import argparse
import time
import os
from distutils.dir_util import copy_tree

import lasagne
from lasagne.regularization import apply_penalty, l2
import numpy as np
import theano
from theano import config
import theano.tensor as T

import mainloop_helpers as mlh
import model_helpers as mh

print ("config floatX: {}".format(config.floatX))

import getpass
CLUSTER = getpass.getuser() in ["tisu32"]

# Main program
def execute(dataset, n_hidden_u, n_hidden_t_enc, n_hidden_t_dec, n_hidden_s,
            embedding_source=None,
            num_epochs=500, learning_rate=.001, learning_rate_annealing=1.0,
            alpha=1, beta=1, gamma=1, lmd=.0001, disc_nonlinearity="sigmoid",
            encoder_net_init=0.2, decoder_net_init=0.2, optimizer="rmsprop",
            max_patience=100, batchnorm=0, keep_labels=1.0,
            prec_recall_cutoff=True, missing_labels_val=-1.0, which_fold=0,
            early_stop_criterion='loss_sup_det',
            save_path='/Tmp/romerosa/DietNetworks/newmodel/',
            save_copy='/Tmp/romerosa/DietNetworks/',
            dataset_path='/Tmp/' + os.environ["USER"] + '/datasets/',
            resume=False, exp_name='', random_proj=0):

    # Prepare embedding information
    if embedding_source is None or embedding_source == "raw":
        embedding_source = None
        embedding_input = 'raw'
    elif os.path.exists(embedding_source):
        embedding_input = embedding_source
    else:
        embedding_input = embedding_source
        embedding_source = os.path.join(dataset_path, embedding_input + '_fold' + str(which_fold) + '.npy')

    # Load the dataset
    print("Loading data")
    x_train, y_train, x_valid, y_valid, x_test, y_test, \
        x_unsup, training_labels = mlh.load_data(
            dataset, dataset_path, embedding_source,
            which_fold=which_fold, keep_labels=keep_labels,
            missing_labels_val=missing_labels_val,
            embedding_input=embedding_input)

    if x_unsup is not None:
        n_samples_unsup = x_unsup.shape[1]
    else:
        n_samples_unsup = 0

    # Extract required information from data
    n_samples, n_feats = x_train.shape
    print("Number of features : ", n_feats)
    print("Glorot init : ", 2.0 / (n_feats + n_hidden_t_enc[-1]))
    n_targets = y_train.shape[1]

    # Set some variables
    batch_size = 128
    beta = gamma if (gamma == 0) else beta

    # Preparing folder to save stuff
    if embedding_source is None:
        embedding_name = embedding_input
    else:
        embedding_name = embedding_source.replace("_", "").split(".")[0]
        exp_name += embedding_name.rsplit('/', 1)[::-1][0] + '_'

    exp_name += mlh.define_exp_name(keep_labels, alpha, beta, gamma, lmd,
                                    n_hidden_u, n_hidden_t_enc, n_hidden_t_dec,
                                    n_hidden_s, which_fold,
                                    learning_rate, decoder_net_init,
                                    encoder_net_init, early_stop_criterion,
                                    learning_rate_annealing)

    print("Experiment: " + exp_name)
    save_path = os.path.join(save_path, dataset, exp_name)
    save_copy = os.path.join(save_copy, dataset, exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_copy):
        os.makedirs(save_copy)

    # Prepare Theano variables for inputs and targets
    input_var_sup = T.matrix('input_sup')
    input_var_unsup = theano.shared(x_unsup, 'input_unsup')  # x_unsup TBD
    target_var_sup = T.matrix('target_sup')
    lr = theano.shared(np.float32(learning_rate), 'learning_rate')

    # Build model
    print("Building model")

    # Some checkings
    # assert len(n_hidden_u) > 0
    assert len(n_hidden_t_enc) > 0
    assert len(n_hidden_t_dec) > 0
    assert n_hidden_t_dec[-1] == n_hidden_t_enc[-1]

    # Build feature embedding networks (encoding and decoding if gamma > 0)
    nets, embeddings, pred_feat_emb = mh.build_feat_emb_nets(
        embedding_source, n_feats, n_samples_unsup,
        input_var_unsup, n_hidden_u, n_hidden_t_enc,
        n_hidden_t_dec, gamma, encoder_net_init,
        decoder_net_init, save_path, random_proj)

    # Build feature embedding reconstruction networks (if alpha > 0, beta > 0)
    nets += mh.build_feat_emb_reconst_nets(
            [alpha, beta], n_samples_unsup, n_hidden_u,
            [n_hidden_t_enc, n_hidden_t_dec],
            nets, [encoder_net_init, decoder_net_init])

    # Supervised network
    discrim_net, hidden_rep = mh.build_discrim_net(
        batch_size, n_feats, input_var_sup, n_hidden_t_enc,
        n_hidden_s, embeddings[0], disc_nonlinearity, n_targets, batchnorm)

    # Reconstruct network
    nets += [mh.build_reconst_net(hidden_rep, embeddings[1] if
                                  len(embeddings) > 1
                                  else None, n_feats, gamma)]

    # Load weights if we are resuming job
    if resume:
        # Load best model
        with np.load(os.path.join(save_copy, 'dietnet_last.npz')) as f:
            param_values = [f['arr_%d' % i]
                            for i in range(len(f.files))]
        nlayers = len(lasagne.layers.get_all_params(filter(None, nets) +
                                                    [discrim_net]))
        lasagne.layers.set_all_param_values(filter(None, nets) +
                                            [discrim_net],
                                            param_values[:nlayers])

    print("Building and compiling training functions")

    # Build and compile training functions
    predictions, predictions_det = mh.define_predictions(nets, start=2)
    prediction_sup, prediction_sup_det = mh.define_predictions([discrim_net])
    prediction_sup = prediction_sup[0]
    prediction_sup_det = prediction_sup_det[0]

    # Define losses
    # reconstruction losses
    reconst_losses, reconst_losses_det = mh.define_reconst_losses(
        predictions, predictions_det, [input_var_unsup, input_var_unsup,
                                       input_var_sup])
    # supervised loss
    sup_loss, sup_loss_det = mh.define_sup_loss(
        disc_nonlinearity, prediction_sup, prediction_sup_det, keep_labels,
        target_var_sup, missing_labels_val)

    # Define inputs
    inputs = [input_var_sup, target_var_sup]

    """
    # Define parameters
    params = lasagne.layers.get_all_params(
        [discrim_net]+filter(None, nets), trainable=True)
    params_to_freeze= \
        lasagne.layers.get_all_params(filter(None, nets), trainable=False)
    """

    # Define parameters
    params = lasagne.layers.get_all_params(
        [discrim_net]+list(filter(None, nets)), trainable=True, unwrap_shared=False)
    params_to_freeze= \
        lasagne.layers.get_all_params(filter(None, nets), trainable=False,
                                      unwrap_shared=False)

    # Remove unshared variables from params and params_to_freeze
    params = [p for p in params if isinstance(p, theano.compile.sharedvalue.SharedVariable)]
    params_to_freeze = [p for p in params_to_freeze if isinstance(p, theano.compile.sharedvalue.SharedVariable)]
    print("Params : ", params)

    feat_emb_var = next(p for p in lasagne.layers.get_all_params([discrim_net]) if p.name == 'input_unsup' or p.name == 'feat_emb')
    # feat_emb_var = lasagne.layers.get_all_params([discrim_net])[0]
    print(feat_emb_var)
    feat_emb_val = feat_emb_var.get_value()
    feat_emb_norms = (feat_emb_val ** 2).sum(0) ** 0.5
    feat_emb_var.set_value(feat_emb_val / feat_emb_norms)

    print('Number of params discrim: '+str(len(params)))
    print('Number of params to freeze: '+str(len(params_to_freeze)))

    for p in params_to_freeze:
        new_params = [el for el in params if el != p]
        params = new_params

    print('Number of params to update: '+ str(len(params)))

    # Combine losses
    loss = sup_loss + alpha*reconst_losses[0] + beta*reconst_losses[1] + \
        gamma*reconst_losses[2]
    loss_det = sup_loss_det + alpha*reconst_losses_det[0] + \
        beta*reconst_losses_det[1] + gamma*reconst_losses_det[2]

    l2_penalty = apply_penalty(params, l2)
    loss = loss + lmd*l2_penalty
    loss_det = loss_det + lmd*l2_penalty

    # Compute network updates
    assert optimizer in ["rmsprop", "adam"]
    if optimizer == "rmsprop":
        updates = lasagne.updates.rmsprop(loss,
                                        params,
                                        learning_rate=lr)
    elif optimizer == "adam":
        updates = lasagne.updates.adam(loss,
                                       params,
                                       learning_rate=lr)
    # updates = lasagne.updates.sgd(loss,
    #                               params,
    #                               learning_rate=lr)
    # updates = lasagne.updates.momentum(loss, params,
    #                                    learning_rate=lr, momentum=0.0)

    # Apply norm constraints on the weights
    for k in updates.keys():
        if updates[k].ndim == 2:
            updates[k] = lasagne.updates.norm_constraint(updates[k], 1.0)

    # Compile training function
    train_fn = theano.function(inputs, loss, updates=updates,
                               on_unused_input='ignore')

    # Monitoring Labels
    monitor_labels = ["reconst. feat. W_enc",
                      "reconst. feat. W_dec",
                      "reconst. loss"]
    monitor_labels = [i for i, j in zip(monitor_labels, reconst_losses)
                      if j != 0]
    monitor_labels += ["feat. W_enc. mean", "feat. W_enc var"]
    monitor_labels += ["feat. W_dec. mean", "feat. W_dec var"] if \
        (embeddings[1] is not None) else []
    monitor_labels += ["loss. sup.", "total loss"]

    # Build and compile test function
    val_outputs = reconst_losses_det
    val_outputs = [i for i, j in zip(val_outputs, reconst_losses) if j != 0]
    val_outputs += [embeddings[0].mean(), embeddings[0].var()]
    val_outputs += [embeddings[1].mean(), embeddings[1].var()] if \
        (embeddings[1] is not None) else []
    val_outputs += [sup_loss_det, loss_det]

    # Compute accuracy and add it to monitoring list
    test_acc, test_pred = mh.define_test_functions(
        disc_nonlinearity, prediction_sup, prediction_sup_det, target_var_sup)
    monitor_labels.append("accuracy")
    val_outputs.append(test_acc)

    # Compile prediction function
    predict = theano.function([input_var_sup], test_pred)

    # Compile validation function
    val_fn = theano.function(inputs,
                             [prediction_sup_det] + val_outputs,
                             on_unused_input='ignore')

    # Finally, launch the training loop.
    print("Starting training...")

    # Some variables
    patience = 0

    train_monitored = []
    valid_monitored = []
    train_loss = []

    # Pre-training monitoring
    print("Epoch 0 of {}".format(num_epochs))

    train_minibatches = mlh.iterate_minibatches(x_train, y_train,
                                                batch_size, shuffle=False)
    train_err = mlh.monitoring(train_minibatches, "train", val_fn,
                               monitor_labels, prec_recall_cutoff)

    valid_minibatches = mlh.iterate_minibatches(x_valid, y_valid,
                                                batch_size, shuffle=False)
    valid_err = mlh.monitoring(valid_minibatches, "valid", val_fn,
                               monitor_labels, prec_recall_cutoff)

    # Training loop
    start_training = time.time()
    for epoch in range(num_epochs):
        start_time = time.time()
        print("Epoch {} of {}".format(epoch+1, num_epochs))
        nb_minibatches = 0
        loss_epoch = 0

        # Train pass
        for batch in mlh.iterate_minibatches(x_train, training_labels,
                                             batch_size,
                                             shuffle=True):
            loss_epoch += train_fn(*batch)
            nb_minibatches += 1

        loss_epoch /= nb_minibatches
        train_loss += [loss_epoch]

        # Monitoring on the training set
        train_minibatches = mlh.iterate_minibatches(x_train, y_train,
                                                    batch_size, shuffle=False)
        train_err = mlh.monitoring(train_minibatches, "train", val_fn,
                                   monitor_labels, prec_recall_cutoff)
        train_monitored += [train_err]

        # Monitoring on the validation set
        valid_minibatches = mlh.iterate_minibatches(x_valid, y_valid,
                                                    batch_size, shuffle=False)

        valid_err = mlh.monitoring(valid_minibatches, "valid", val_fn,
                                   monitor_labels, prec_recall_cutoff)
        valid_monitored += [valid_err]

        try:
            early_stop_val = valid_err[
                monitor_labels.index(early_stop_criterion)]
        except:
            raise ValueError("There is no monitored value by the name of %s" %
                             early_stop_criterion)

        valid_loss_sup_hist = [v[monitor_labels.index("loss. sup.")] for v in valid_monitored]
        valid_loss_sup = valid_loss_sup_hist[-1]

        # Early stopping
        if epoch == 0:
            best_valid = early_stop_val
        elif ((early_stop_val > best_valid and early_stop_criterion == 'accuracy') or
              (early_stop_val >= best_valid and early_stop_criterion == 'accuracy' and
               valid_loss_sup == min(valid_loss_sup_hist)) or
              (early_stop_val < best_valid and early_stop_criterion == 'loss. sup.')):
            best_valid = early_stop_val
            patience = 0

            # Save stuff
            np.savez(os.path.join(save_path, 'dietnet_best.npz'),
                     *lasagne.layers.get_all_param_values(list(filter(None, nets)) +
                                                          [discrim_net]))
            np.savez(save_path + "/errors_supervised_best.npz",
                     zip(*train_monitored), zip(*valid_monitored))

            # Monitor on the test set now because sometimes the saving doesn't
            # go well and there isn't a model to load at the end of training
            if y_test is not None:
                test_minibatches = mlh.iterate_minibatches(x_test, y_test,
                                                           138,
                                                           shuffle=False)

                test_err = mlh.monitoring(test_minibatches, "test", val_fn,
                                          monitor_labels, prec_recall_cutoff)
        else:
            patience += 1
            # Save stuff
            np.savez(os.path.join(save_path, 'dietnet_last.npz'),
                     *lasagne.layers.get_all_param_values(list(filter(None, nets)) +
                                                          [discrim_net]))
            np.savez(save_path + "/errors_supervised_last.npz",
                     zip(*train_monitored), zip(*valid_monitored))

        # End training
        if patience == max_patience or epoch == num_epochs-1:
            print("Ending training")
            # Load best model
            with np.load(os.path.join(save_path, 'dietnet_best.npz')) as f:
                param_values = [f['arr_%d' % i]
                                for i in range(len(f.files))]
            nlayers = len(lasagne.layers.get_all_params(list(filter(None, nets)) +
                                                        [discrim_net]))
            lasagne.layers.set_all_param_values(list(filter(None, nets)) +
                                                [discrim_net],
                                                param_values[:nlayers])
            if embedding_source is None:
                # Save embedding
                pred = pred_feat_emb()
                np.savez(os.path.join(save_path, 'feature_embedding.npz'),
                         pred)

            # Training set results
            train_minibatches = mlh.iterate_minibatches(x_train, y_train,
                                                        batch_size,
                                                        shuffle=False)
            train_err = mlh.monitoring(train_minibatches, "train", val_fn,
                                       monitor_labels, prec_recall_cutoff)

            # Validation set results
            valid_minibatches = mlh.iterate_minibatches(x_valid, y_valid,
                                                        batch_size,
                                                        shuffle=False)
            valid_err = mlh.monitoring(valid_minibatches, "valid", val_fn,
                                       monitor_labels, prec_recall_cutoff)

            # Test set results
            if y_test is not None:
                test_minibatches = mlh.iterate_minibatches(x_test, y_test,
                                                           138,
                                                           shuffle=False)

                test_err = mlh.monitoring(test_minibatches, "test", val_fn,
                                          monitor_labels, prec_recall_cutoff)
            else:
                for minibatch in mlh.iterate_testbatches(x_test,
                                                         138,
                                                         shuffle=False):
                    test_predictions = []
                    test_predictions += [predict(minibatch)]
                np.savez(os.path.join(save_path, 'test_predictions.npz'),
                         test_predictions)

            # Stop
            print("  epoch time:\t\t\t{:.3f}s \n".format(time.time() -
                                                         start_time))
            break

        print("  epoch time:\t\t\t{:.3f}s \n".format(time.time() - start_time))

        # Anneal the learning rate
        lr.set_value(float(lr.get_value() * learning_rate_annealing))

    # Print all final errors for train, validation and test
    print("Training time:\t\t\t{:.3f}s".format(time.time() - start_training))

    # Copy files to loadpath
    if save_path != save_copy:
        print('Copying model and other training files to {}'.format(save_copy))
        copy_tree(save_path, save_copy)


def main():
    parser = argparse.ArgumentParser(description="""Train Diet Networks""")
    parser.add_argument('--dataset',
                        default='1000_genomes',
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
                        default='histo3x26',
                        help='Source for the feature embedding. Either' +
                             'None or the name of a file from which' +
                             'to load a learned embedding')
    parser.add_argument('--num_epochs',
                        '-ne',
                        type=int,
                        default=500,
                        help="""Int to indicate the max'
                        'number of epochs.""")
    parser.add_argument('--learning_rate',
                        '-lr',
                        type=float,
                        default=0.0001,
                        help="""Float to indicate learning rate.""")
    parser.add_argument('--learning_rate_annealing',
                        '-lra',
                        type=float,
                        default=.99,
                        help="Float to indicate learning rate annealing rate.")
    parser.add_argument('--alpha',
                        '-a',
                        type=float,
                        default=0.,
                        help="""reconst_loss coeff. for auxiliary net W_enc""")
    parser.add_argument('--beta',
                        '-b',
                        type=float,
                        default=0.,
                        help="""reconst_loss coeff. for auxiliary net W_dec""")
    parser.add_argument('--gamma',
                        '-g',
                        type=float,
                        default=10.,
                        help="""reconst_loss coeff. (used for aux net W-dec as well)""")
    parser.add_argument('--lmd',
                        '-l',
                        type=float,
                        default=.0,
                        help="""Weight decay coeff.""")
    parser.add_argument('--disc_nonlinearity',
                        '-nl',
                        default="softmax",
                        help="""Nonlinearity to use in disc_net's last layer""")
    parser.add_argument('--encoder_net_init',
                        '-eni',
                        type=float,
                        default=0.01,
                        help="Bounds of uniform initialization for " +
                             "encoder_net weights")
    parser.add_argument('--decoder_net_init',
                        '-dni',
                        type=float,
                        default=0.01,
                        help="Bounds of uniform initialization for " +
                             "decoder_net weights")
    parser.add_argument('--optimizer',
                        type=str,
                        default="rmsprop",
                        help="Optimizer to use for training (rmsprop or adam)")
    parser.add_argument('--patience',
                        type=int,
                        default=100,
                        help="Number of epochs without validation improvement" +
                             "after which to stop training")
    parser.add_argument('--batchnorm',
                        '-bn',
                        type=int,
                        default=0,
                        help="Whether to use BatchNorm in the main network")
    parser.add_argument('--keep_labels',
                        type=float,
                        default=1.0,
                        help='Fraction of training labels to keep')
    parser.add_argument('--prec_recall_cutoff',
                        type=int,
                        default=0,
                        help='Whether to compute the precision-recall cutoff' +
                             'or not')
    parser.add_argument('--which_fold',
                        type=int,
                        default=0,
                        help='Which fold to use for cross-validation (0-4)')
    parser.add_argument('--early_stop_criterion',
                        default='accuracy',
                        help='What monitored variable to use for early-stopping')
    parser.add_argument('--save_tmp',
                        default= '/Tmp/'+ os.environ["USER"]+'/DietNetworks/' if not CLUSTER else
                            '$SCRATCH'+'/DietNetworks/',
                        help='Path to save results.')
    parser.add_argument('--save_perm',
                        default='/data/lisatmp4/'+ os.environ["USER"]+'/DietNetworks/' if not CLUSTER else
                            '$SCRATCH'+'/DietNetworks/',
                        help='Path to save results.')
    parser.add_argument('--dataset_path',
                        default='/data/lisatmp4/romerosa/datasets/1000_Genome_project/data_files/300k/',
                        help='Path to dataset')
    parser.add_argument('-resume',
                        type=bool,
                        default=False,
                        help='Whether to resume job')
    parser.add_argument('-exp_name',
                        type=str,
                        default='dietnets_final_',
                        help='Experiment name that will be concatenated at the beginning of the generated name')
    parser.add_argument('--random_proj',
                        '-rp',
                        type=int,
                        default=0,
                        help="Whether to use random projections as embedding")

    args = parser.parse_args()
    print ("Printing args")
    print (args)

    execute(args.dataset,
            mlh.parse_int_list_arg(args.n_hidden_u),
            mlh.parse_int_list_arg(args.n_hidden_t_enc),
            mlh.parse_int_list_arg(args.n_hidden_t_dec),
            mlh.parse_int_list_arg(args.n_hidden_s),
            args.embedding_source,
            int(args.num_epochs),
            args.learning_rate,
            args.learning_rate_annealing,
            args.alpha,
            args.beta,
            args.gamma,
            args.lmd,
            args.disc_nonlinearity,
            args.encoder_net_init,
            args.decoder_net_init,
            args.optimizer,
            args.patience,
            args.batchnorm,
            args.keep_labels,
            args.prec_recall_cutoff != 0, -1,
            args.which_fold,
            args.early_stop_criterion,
            args.save_tmp,
            args.save_perm,
            args.dataset_path,
            args.resume,
            args.exp_name,
            int(args.random_proj))


if __name__ == '__main__':
    main()
