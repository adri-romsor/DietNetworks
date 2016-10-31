from __future__ import print_function
import argparse
import time
import os
import tables
import scipy
import ipdb
import lasagne
from lasagne.layers import DenseLayer, InputLayer
from lasagne.nonlinearities import sigmoid, softmax, tanh, linear, rectify
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from epls import EPLS, tensor_fun_EPLS
from feature_selection.experiments.common import dataset_utils, imdb


def iterate_minibatches(x, batch_size, shuffle=False, dataset=None):
    indices = np.arange(x.shape[0])
    if shuffle:
        indices = np.random.permutation(x.shape[0])
    for i in range(0, x.shape[0]-batch_size+1, batch_size):
        yield x[indices[i:i+batch_size], :]


def iterate_sparse_minibatches(x, batch_size, shuffle=False, dataset=None):
    indices = np.arange(x.shape[0])
    if shuffle:
        indices = np.random.permutation(x.shape[0])
    for i in range(0, x.shape[0]-batch_size+1, batch_size):
        yield x[indices[i:i+batch_size], :].toarray().astype("float32")


def monitoring(minibatches, which_set, error_fn, monitoring_labels):
    print('-'*20 + which_set + ' monit.' + '-'*20)
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


def exp_name(n_hidden_u, num_epochs, l_r, ngram_range, use_unlab_for_dic,
             use_unlab_for_emb):
    # Define experiment name from parameters
    exp_name = 'pretrain_embedding_AE_ngram_1'+str(ngram_range[1]) + '_' + \
                str(num_epochs) + 'epochs_lr-' + str(l_r) + \
                ('no' if not use_unlab_for_dic else '') + '_unlab-in-dic' + \
                ('no' if not use_unlab_for_dic else '') + \
                '_unlab-in-pretraining-set'

    exp_name += '_hu'
    for i in range(len(n_hidden_u)):
        exp_name += ("-" + str(n_hidden_u[i]))

    return exp_name


# Main program
def execute(dataset, n_hidden_u, num_epochs=500,
            learning_rate=.001, ngram_range=(1, 1), use_unlab_for_dic=False,
            use_unlab_for_emb=False,
            save_path='/Tmp/$USER/feature_selection/newmodel/'):

    # Load the dataset
    print("Loading data")
    splits = [0.8]  # This will split the data into [80%, 20%]
    if dataset == 'protein_binding':
        data = dataset_utils.load_protein_binding(transpose=True,
                                                  splits=splits)
    elif dataset == 'dorothea':
        data = dataset_utils.load_dorothea(transpose=True, splits=splits)
    elif dataset == 'opensnp':
        data = dataset_utils.load_opensnp(transpose=True, splits=splits)
    elif dataset == 'reuters':
        data = dataset_utils.load_reuters(transpose=True, splits=splits)
    elif dataset == 'imdb':
        # data = dataset_utils.load_imdb(transpose=True, splits=splits)
        # ipdb.set_trace()
        mes = 'Loading 1'
        if ngram_range[1] != '1':
            mes += '&2' if ngram_range[1] == '2' else '&2&3'
        mes += '-gram data'
        print(mes)
        train_data, _, unlab_data, \
            _, _, _ = imdb.load_imdb(path='/Tmp/erraqaba/'
                                          'data/imdb/',
                                     use_unlab=use_unlab_for_dic,
                                     ngram_range=ngram_range)
        if use_unlab_for_emb:
            data = scipy.sparse.vstack([train_data, unlab_data]).transpose()
        else:
            data = train_data.transpose()
            # data = imdb.read_from_hdf5(unsupervised=True, feat_type='BoW')
        split_index = int(data.shape[0]*splits[0])
    elif dataset == 'dragonn':
        from feature_selection.experiments.common import dragonn_data
        data = dragonn_data.load_data(500, 10000, 10000)
    else:
        print("Unknown dataset")
        return

    if dataset == 'imdb':
        x_train = data[:split_index]
        x_valid = data[split_index:]
    else:
        x_train = data[0][0]
        x_valid = data[1][0]

    # Extract required information from data
    n_row, n_col = x_train.shape

    # Set some variables
    batch_size = 256

    # Preparing folder to save stuff
    experim_name = exp_name(n_hidden_u, num_epochs, learning_rate,
                            ngram_range, use_unlab_for_dic, use_unlab_for_emb)

    save_path = save_path + dataset + '_' + experim_name + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('input_unsup')
    lr = theano.shared(np.float32(learning_rate), 'learning_rate')

    # Build model
    print("Building model")
    # Some checkings
    assert len(n_hidden_u) > 0
    # Build network
    # encoder
    encoder_net = {}
    lname = ['enc_0']
    encoder_net[lname[0]] = InputLayer((batch_size, n_col), input_var)
    for i, out in enumerate(n_hidden_u):
        lname.append('enc_'+str(i+1))
        encoder_net[lname[i+1]] = DenseLayer(encoder_net[lname[i]],
                                             num_units=out,
                                             nonlinearity=tanh)
    feat_emb = lasagne.layers.get_output(encoder_net[lname[-1]])
    pred_feat_emb = theano.function([input_var], feat_emb)
    # decoder
    decoder_net = encoder_net[lname[-1]]
    for i in range(len(n_hidden_u)-2, -1, -1):
        decoder_net = DenseLayer(decoder_net, num_units=n_hidden_u[i],
                                 nonlinearity=tanh)
    decoder_net = DenseLayer(decoder_net, num_units=n_col,
                             nonlinearity=rectify)

    print("Building and compiling training functions")
    # Some variables
    params = []
    # Build and compile training functions
    # Unsupervised reconstruction functions
    reconstruction = lasagne.layers.get_output(decoder_net)
    reconstruction_det = lasagne.layers.get_output(decoder_net,
                                                   deterministic=True)

    # output_recon = theano.function([input_var], reconstruction)

    loss = lasagne.objectives.squared_error(
        reconstruction,
        input_var).mean()

    # RE = theano.function([input_var], loss)

    loss_det = lasagne.objectives.squared_error(
        reconstruction_det,
        input_var).mean()

    params += lasagne.layers.get_all_params(decoder_net, trainable=True)

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
    train_fn = theano.function([input_var], loss,
                               updates=updates,
                               on_unused_input='ignore')

    # Monitoring Values
    monitor_labels = ['total_loss_det']
    val_outputs = [loss_det]

    # Add some monitoring on the learned feature embedding
    # val_outputs += [feat_emb.min(), feat_emb.mean(),
    #                 feat_emb.max(), feat_emb.var()]
    # monitor_labels += ["feat. emb. min", "feat. emb. mean",
    #                    "feat. emb. max", "feat. emb. var"]

    # Compile validation function
    val_fn = theano.function([input_var],
                             val_outputs)

    # Finally, launch the training loop.
    print("Starting training...")

    # Some variables
    max_patience = 100
    patience = 0

    train_loss = []
    train_curve = []
    valid_loss = []

    nb_minibatches = n_row/batch_size
    start_training = time.time()
    train_minibatches = iterate_sparse_minibatches(x_train, batch_size,
                                                   dataset=dataset,
                                                   shuffle=True)
    for epoch in range(num_epochs):
        start_time = time.time()
        print("Epoch {} of {}".format(epoch+1, num_epochs))

        loss_epoch = 0
        # count = 0
        # Train pass
        for batch in iterate_sparse_minibatches(x_train, batch_size,
                                                dataset=dataset, shuffle=True):
            loss_epoch += train_fn(batch)

        loss_epoch /= nb_minibatches
        train_loss += [loss_epoch]
        train_minibatches = iterate_sparse_minibatches(x_train, batch_size,
                                                       dataset=dataset,
                                                       shuffle=True)
        train_err = monitoring(train_minibatches, "train", val_fn,
                               monitor_labels)
        # if epoch == 20:
        #     ipdb.set_trace()
        train_curve += [train_err[0]]
        # Validation pass
        valid_minibatches = iterate_sparse_minibatches(x_valid, batch_size,
                                                       dataset=dataset,
                                                       shuffle=True)

        valid_err = monitoring(valid_minibatches, "valid", val_fn,
                               monitor_labels)

        valid_loss += [valid_err[0]]
        # monitor_learning curves
        if epoch % 25 == 0:
            np.savez(save_path+'learning_curves.npz',
                     train_curve=train_curve, valid_loss=valid_loss)

        # Early stopping
        if epoch == 0:
            best_valid = valid_loss[epoch]
        elif valid_loss[epoch] < best_valid:
            best_valid = valid_loss[epoch]
            patience = 0

            # Save stuff
            np.savez(save_path+'model_AE.npz',
                     *lasagne.layers.get_all_param_values(encoder_net[lname[-1]]))
        else:
            patience += 1

        # End training
        if patience == max_patience or epoch == num_epochs-1:
            print("   Ending training")
            # Load unsupervised best model
            with np.load(save_path + 'model_AE.npz',) as f:
                param_values = [f['arr_%d' % i]
                                for i in range(len(f.files))]
                nlayers = len(lasagne.layers.get_all_params(encoder_net[lname[-1]]))
                lasagne.layers.set_all_param_values(encoder_net[lname[-1]],
                                                    param_values[:nlayers])
            # Save learning curves
            np.savez(save_path+'learning_curves.npz', train_curve=train_curve,
                     valid_loss=valid_loss)
            # Save embedding
            preds = []
            for batch in iterate_sparse_minibatches(x_train, 1,
                                                    dataset=dataset,
                                                    shuffle=False):
                preds.append(pred_feat_emb(batch))
            for batch in iterate_sparse_minibatches(x_valid, 1,
                                                    dataset=dataset,
                                                    shuffle=False):
                preds.append(pred_feat_emb(batch))
            preds = np.vstack(preds)
            np.savez(save_path+'feature_embedding.npz', preds)

            # Stop
            print(" epoch time:\t\t\t{:.3f}s".format(time.time() - start_time))
            break
        else:
            print(" epoch time:\t\t\t{:.3f}s".format(time.time() - start_time))

    # Print all final errors for train, validation and test
    print("Training time:\t\t\t{:.3f}s".format(time.time() - start_training))


def parse_int_list_arg(arg):
    if isinstance(arg, str):
        arg = eval(arg)

    if isinstance(arg, list):
        return arg
    if isinstance(arg, int):
        return [arg]
    else:
        raise ValueError("Following arg value could not be cast as a list of"
                         "integer values : " % arg)


def main():
    parser = argparse.ArgumentParser(description="""Implementation of the
                                     feature selection v4""")
    parser.add_argument('--dataset',
                        default='imdb',
                        help='Dataset.')
    parser.add_argument('--n_hidden_u',
                        default=[50],
                        help='List of unsupervised hidden units.')
    parser.add_argument('--num_epochs',
                        '-ne',
                        type=int,
                        default=5,
                        help="""Int to indicate the max'
                        'number of epochs.""")
    parser.add_argument('--learning_rate',
                        '-lr',
                        type=float,
                        default=0.05,
                        help="""Float to indicate learning rate.""")
    parser.add_argument('--ngram_range',
                        default=(1, 1),
                        help="""Float to indicate learning rate.""")
    parser.add_argument('--use_unlab_for_dic',
                        default=True,
                        help='choose to use unlabeled data or not to generate'
                             'dictonary')
    parser.add_argument('--use_unlab_for_emb',
                        default=True,
                        help='choose to use unlabeled data or not to learn'
                             'the embedding')
    parser.add_argument('--save',
                        default='/Tmp/erraqaba/feature_selection/' +
                                'feat_embeddings/',
                        help='Path to save results.')

    args = parser.parse_args()

    execute(args.dataset,
            parse_int_list_arg(args.n_hidden_u),
            int(args.num_epochs),
            args.learning_rate,
            args.ngram_range,
            args.use_unlab_for_dic,
            args.use_unlab_for_emb,
            args.save)


if __name__ == '__main__':
    main()
