from __future__ import print_function
import argparse
import time
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plot

import lasagne
from lasagne.layers import DenseLayer, InputLayer, BatchNormLayer
from lasagne.nonlinearities import (sigmoid, softmax, tanh, linear, rectify,
                                    leaky_rectify, very_leaky_rectify)
import numpy as np
import theano
import theano.tensor as T

#theano.config.compute_test_value = 'warn'

"""
NOTE : This script should be launched with the following Theano flags :
THEANO_FLAGS="floatX=float32,optimizer_excluding=scanOp_pushout_seqs_ops:\
scanOp_pushout_nonseqs_ops:scanOp_pushout_output" python featsel_v3.py \
opensnp [...]
"""

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    targets = targets.transpose()

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


def monitoring(minibatches, dataset_name, val_fn, monitoring_labels):

    monitoring_values = np.zeros(len(monitoring_labels), dtype="float32")
    monitoring_dict = {}
    global_batches = 0

    for batch in minibatches:
        inputs, targets = batch

        # Update monitored values
        out = val_fn(inputs, targets)
        monitoring_values = monitoring_values + out
        global_batches += 1

    # Print monitored values
    monitoring_values /= global_batches
    for (label, val) in zip(monitoring_labels, monitoring_values):
        print ("  {}\t{}:\t\t{:.6f}".format(dataset_name, label, val))
        monitoring_dict[label] = val

    return monitoring_dict


def weights(n_in, n_out, name=None, init=0.01):
    return theano.shared(np.random.uniform(-init, init,
                                           size=(n_in, n_out)).astype("float32"),
                         name)


def biases(n_out, name=None):
    return theano.shared(np.zeros((n_out,), "float32"), name)


# Main program
def execute(dataset, n_output, num_epochs=500, save_path=None,
            cross_validation=None):
    # Load the dataset
    print("Loading data")
    if dataset == 'genomics':
        from feature_selection.experiments.common.dorothea import load_data
        x_train, y_train = load_data('train', 'standard', False, 'numpy')
        x_valid, y_valid = load_data('valid', 'standard', False, 'numpy')

        # WARNING : The dorothea dataset has no test labels
        print ("Warning: using validation as test set in execute, for Dorothea")

        #x_test = load_data('test', 'standard', False, 'numpy')
        #y_test = None

        x_test,y_test = x_valid,y_valid

    elif dataset == 'genomics_all':
        from feature_selection.experiments.common.dorothea import load_data
        x_train, y_train = load_data('all', 'standard', False, 'numpy')
        x_valid = None
        y_valid = None
        x_test = None
        y_test = None

    elif dataset == 'debug':
        x_train = np.random.rand(10, 100).astype(np.float32)
        x_valid = np.random.rand(2, 100).astype(np.float32)
        x_test = np.random.rand(2, 100).astype(np.float32)
        y_train = np.random.randint(0, 2, size=10).astype('int32')
        y_valid = np.random.randint(0, 2, size=2).astype('int32')
        y_test = np.random.randint(0, 2, size=2).astype('int32')

    elif dataset == 'debug_snp':
        # Creates a debug dataset meant to look like openSNP so there is
        # something to test this code on until openSNP has been successfully
        # wrapped and integrated

        n_train = 1000
        n_valid = 500
        n_test = 500
        n_features = 1000000

        x_train = np.random.randint(0, 20, size=(n_train,
                                                 n_features)).astype('int8')
        x_valid = np.random.randint(0, 20, size=(n_valid,
                                                 n_features)).astype('int8')
        x_test = np.random.randint(0, 20, size=(n_test,
                                                n_features)).astype('int8')

        y_train = np.random.randint(0, 200, size=n_train).astype('int8')
        y_valid = np.random.randint(0, 200, size=n_valid).astype('int8')
        y_test = np.random.randint(0, 200, size=n_test).astype('int8')

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
    
    # With the dataset loaded, check if cross-validation needs to be applied.
    if cross_validation is not None:
        
        no_fold, nb_folds = cross_validation
        
        assert no_fold >= 1 and no_fold <= nb_folds
        assert nb_folds >= 2
        
        full_x = np.vstack((x_train, x_valid))
        full_y = np.hstack((y_train, y_valid))
        
        start_idx = ((len(full_y) + nb_folds - 1) / nb_folds) * (no_fold - 1)
        end_idx = ((len(full_y) + nb_folds - 1) / nb_folds) * (no_fold)
        
        train_examples = range(0, start_idx) + range(end_idx, len(full_y))
        valid_examples = range(start_idx, min(end_idx, len(full_y)))
        
        train_x = full_x[train_examples]
        train_y = full_y[train_examples]
        valid_x = full_x[valid_examples]
        valid_y = full_y[valid_examples]
        
        print("Altering training and validation datasets to perform k-fold"
              "crossvalidation for fold %i out of %i." % (no_fold, nb_folds))
        print("New train shape :", train_x.shape)
        print("New valid shape :", valid_x.shape)

    # supervised_type refers to the parameters of the network, and
    # changes the last layer.
    if dataset in ["opensnp", "debug", "debug_snp"]:
        supervised_type = "supervised_regression"
    elif dataset in ["genomics","genomics_all"]:
        supervised_type = "supervised_classification"
    else:
        print (dataset)
        raise NotImplementedError()

    n_samples, n_feats = x_train.shape
    n_classes = y_train.max() + 1
    n_batch = 10

    # Prepare Theano variables for inputs and targets
    input_var = T.fmatrix('inputs')
    target_var = T.fvector('targets')

    feature_var = theano.shared(x_train.transpose().astype("float32"), 'feature_var')
    lr = theano.shared(np.float32(1e-1), 'learning_rate')

    #input_var.tag.test_value = x_train[:20]
    #target_var.tag.test_value = y_train[:20]
    #feature_var.tag.test_value = x_train.transpose()

    # Build model
    print("Building model")

    # Define a few useful values for building the model
    feat_repr_size = 5
    h_rep_size = 5

    # Define the network architecture
    # NOTE : Can scale down the param init from 0.001 to 0.0001 for slightly
    # better results.
    params = {
        'feature': {
            'layers': [feat_repr_size],
            'weights': [weights(n_samples, feat_repr_size, 'fw1', 0.001)],
            'biases': [biases(feat_repr_size, 'fb1')],
            'acts': [rectify]},
        'encoder_w': {
            'layers': [n_output, h_rep_size],
            'weights': [weights(feat_repr_size, n_output, 'ew1', 0.001),
                        weights(n_output, h_rep_size, 'ew2', 0.0001)],
            'biases': [biases(n_output, 'eb1'),
                       biases(h_rep_size, 'eb2')],
            'acts': [rectify, tanh]},
        'decoder_w': {
            'layers': [n_output, h_rep_size],
            'weights': [weights(feat_repr_size, n_output, 'dw1', 0.001),
                        weights(n_output, h_rep_size, 'dw2', 0.001)],
            'biases': [biases(n_output, 'db1'),
                       biases(h_rep_size, 'db2')],
            'acts': [rectify, tanh]},
        'supervised_regression': {
            'layers': [n_output, 1],
            'weights': [weights(h_rep_size, n_output, 'sw1'),
                        weights(n_output, 1, 'sw2')],
            'biases': [biases(n_output, 'sb1'),
                       biases(1, 'sb2')],
            'acts': [rectify, very_leaky_rectify]},
        'supervised_classification': {
            'layers': [n_output, 1],
            'weights': [weights(h_rep_size, n_output, 'sw1'),
                        weights(n_output, 1, 'sw2')],
            'biases': [biases(n_output, 'sb1'),
                       biases(1, 'sb2')],
            'acts': [rectify, sigmoid]}
            }

    
    params = {
        'feature': {
            'layers': [feat_repr_size],
            'weights': [weights(n_samples, feat_repr_size, 'fw1', 0.0001)],
            'biases': [biases(feat_repr_size, 'fb1')],
            'acts': [rectify]},
        'encoder_w': {
            'layers': [h_rep_size],
            'weights': [weights(feat_repr_size, h_rep_size, 'ew2', 0.0001)],
            'biases': [biases(h_rep_size, 'eb2')],
            'acts': [tanh]},
        'decoder_w': {
            'layers': [h_rep_size],
            'weights': [weights(feat_repr_size, h_rep_size, 'dw2', 0.0001)],
            'biases': [biases(h_rep_size, 'db2')],
            'acts': [tanh]},
        'supervised_regression': {
            'layers': [1],
            'weights': [weights(h_rep_size, 1, 'sw1')],
            'biases': [biases(1, 'sb1')],
            'acts': [very_leaky_rectify]}}
    

    # Build the portion of the model that will predict the encoder's hidden
    # representation fron the inputs and the feature activations.
    def step_enc(dataset, inputs):

        # Predict the feature representations
        feature_rep_net = InputLayer((n_feats, n_samples), dataset)
        for i in range(len(params['feature']['layers'])):
            if i > 0:
                feature_rep_net = BatchNormLayer(feature_rep_net)
            feature_rep_net = DenseLayer(feature_rep_net,
                                num_units=params['feature']['layers'][i],
                                W=params['feature']['weights'][i],
                                b=params['feature']['biases'][i],
                                nonlinearity=params['feature']['acts'][i])

        # Predict the encoder parameters for the features
        enc_weights_net = feature_rep_net
        for i in range(len(params['encoder_w']['layers'])):
            enc_weights_net = BatchNormLayer(enc_weights_net)
            enc_weights_net = DenseLayer(enc_weights_net,
                                num_units=params['encoder_w']['layers'][i],
                                W=params['encoder_w']['weights'][i],
                                b=params['encoder_w']['biases'][i],
                                nonlinearity=params['encoder_w']['acts'][i])
        encoder_weights = lasagne.layers.get_output(enc_weights_net)

        # Predict the contribution to the encoder's hidden representation.
        return T.dot(inputs, encoder_weights)

    results, updates = theano.scan(fn=step_enc,
                                   non_sequences=[feature_var, input_var],
                                   n_steps=1, allow_gc=True,
                                   name="encoder_w_scan")
    hidden_contribution = results[0]

    # Add a bias vector to the hidden_contribution and apply an activation
    # function to obtain the autoencoder's hidden representation
    encoder_b = biases(h_rep_size, 'encoder_b')
    h_rep = T.nnet.relu(hidden_contribution + encoder_b)

    # Build the decoder
    def step_dec(dataset, h_rep):

        # Predict the feature representations
        feature_rep_net = InputLayer((n_feats, n_samples), dataset)
        for i in range(len(params['feature']['layers'])):
            if i > 0:
                feature_rep_net = BatchNormLayer(feature_rep_net)
            feature_rep_net = DenseLayer(feature_rep_net,
                                num_units=params['feature']['layers'][i],
                                W=params['feature']['weights'][i],
                                b=params['feature']['biases'][i],
                                nonlinearity=params['feature']['acts'][i])

        # Predict the decoder parameters for the features
        dec_weights_net = feature_rep_net
        for i in range(len(params['decoder_w']['layers'])):
            dec_weights_net = BatchNormLayer(dec_weights_net)
            dec_weights_net = DenseLayer(dec_weights_net,
                                num_units=params['decoder_w']['layers'][i],
                                W=params['decoder_w']['weights'][i],
                                b=params['decoder_w']['biases'][i],
                                nonlinearity=params['decoder_w']['acts'][i])
        decoder_weights = lasagne.layers.get_output(dec_weights_net).transpose()

        # Predict the contribution to the decoder's reconstruction
        recon_contribution = T.dot(h_rep, decoder_weights)
        return recon_contribution

    results, updates = theano.scan(fn=step_dec,
                                   non_sequences=[feature_var, h_rep],
                                   n_steps=1, allow_gc=True,
                                   name="decoder_w_scan")
    reconstruction = results[0]

    # Build the supervised network that takes the hidden representation of the
    # encoder as input and tries to predict the targets
    supervised_net = InputLayer((n_batch, h_rep_size), h_rep)

    for i in range(len(params[supervised_type]['layers'])):
        supervised_net = DenseLayer(
                supervised_net,
                num_units=params[supervised_type]['layers'][i],
                W=params[supervised_type]['weights'][i],
                b=params[supervised_type]['biases'][i],
                nonlinearity=params[supervised_type]['acts'][i])

    prediction = lasagne.layers.get_output(supervised_net)[:, 0]

    # Compute loss expressions
    print("Defining loss functions")

    unsup_loss = ((input_var - reconstruction) ** 2).mean()

    #unsup_loss = lasagne.objectives.binary_crossentropy(T.nnet.sigmoid(reconstruction) / 2,
    #                                                    input_var / 2).mean()

    if supervised_type == "supervised_regression":
        sup_loss = ((target_var - prediction) ** 2 *
                     T.neq(target_var, -1)).mean()
        sup_abs_loss = (abs(target_var - prediction) *
                        T.neq(target_var, -1)).mean()

    elif supervised_type == "supervised_classification":
        index_0 = T.eq(target_var,0)
        index_1 = T.eq(target_var,1)

        pred_0 = prediction[index_0]
        pred_1 = prediction[index_1]

        prop_0 = T.sum(index_0) / index_0.shape[0]

        sup_loss = - (T.log(pred_0) * (1 - prop_0) + T.log(1 - pred_1) * prop_0)\
                .mean()

    else:
        raise NotImplementedError()

    total_loss = sup_loss

    # Define training funtions
    print("Building training functions")


    # There are 3 places where the parameters are defined:
    # - In the 'params' dictionnary
    # - The encoder biases
    # - The params in the 'supervised_net' network
    all_params = ([encoder_b] +
              params['feature']['weights'] +
              params['feature']['biases'] +
              params['encoder_w']['weights'] +
              params['encoder_w']['biases'] +
              params['decoder_w']['weights'] +
              params['decoder_w']['biases'] +
              params[supervised_type]['weights'] +
              params[supervised_type]['biases'])
    
    all_params = ([encoder_b] +
              params['feature']['weights'] +
              params['feature']['biases'] +
              params['encoder_w']['weights'] +
              params['encoder_w']['biases'] +
              params[supervised_type]['weights'] +
              params[supervised_type]['biases'])

    # Do not train 'feature_var'. It's only a shared variable to avoid
    # transfers to/from the gpu
    all_params = [p for p in all_params if p is not feature_var]

    updates = lasagne.updates.rmsprop(total_loss,
                                      all_params,
                                      learning_rate=lr)
    #updates = lasagne.updates.adam(total_loss, all_params, learning_rate=lr)

    # Reduce lr for parameters of the 'weights' predicting networks
    for p in (params['feature']['weights'] + params['feature']['biases'] +
              params['encoder_w']['weights'] + params['encoder_w']['biases'] +
              params['decoder_w']['weights'] + params['decoder_w']['biases'] +
              params[supervised_type]['weights'] + [encoder_b]):

        if p in updates:
            updates[p] = p + (updates[p] - p) / 10000

    train_fn = theano.function([input_var, target_var], total_loss,
                               updates=updates)

    # Define monitoring functions
    print("Building monitoring functions")

    if supervised_type == "supervised_regression":
        val_fn = theano.function([input_var, target_var],
                             [unsup_loss, sup_loss, sup_abs_loss])
        monitor_labels = ["reconstruction loss", "prediction loss",
                      "prediction loss (MAE)"]
    elif supervised_type == "supervised_classification":
        val_fn = theano.function([input_var, target_var],
                            [unsup_loss, sup_loss])
        monitor_labels = ["reconstruction loss", "prediction loss"]
    else:
        raise NotImplementedError()

    # Finally, launch the training loop.
    print("Starting training...")

    # We iterate over epochs:
    best_valid_mse = 1e20    
    best_epoch = 0
    train_MSEs = []
    valid_MSEs = []
    test_MSE_epochs = []
    test_MSEs = []
    
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data to updates
        # the parameters:
        start_time = time.time()
        for batch in iterate_minibatches(x_train, y_train, n_batch,
                                         shuffle=True):
            inputs, targets = batch
            train_fn(inputs, targets)

        print("Epoch {} of {}".format(epoch + 1, num_epochs))

        # Monitor progress on the training set
        train_minibatches = iterate_minibatches(x_train, y_train, n_batch,
                                                shuffle=False)
        t_mse = monitoring(train_minibatches, "train",
                           val_fn, monitor_labels)['prediction loss']
        train_MSEs.append(t_mse)

        # Monitor progress on the validation set
        valid_minibatches = iterate_minibatches(x_valid, y_valid, n_batch,
                                                shuffle=False)
        v_mse = monitoring(valid_minibatches, "valid",
                           val_fn, monitor_labels)['prediction loss']
        valid_MSEs.append(v_mse)

        # Monitor the test set if needed
        if v_mse < best_valid_mse:
            best_valid_mse = v_mse
            best_epoch = epoch + 1
   
            test_minibatches = iterate_minibatches(x_test, y_test, n_batch,
                                                   shuffle=False)
            t_mse = monitoring(test_minibatches, "test", val_fn,
                               monitor_labels)['prediction loss']
            
            test_MSE_epochs.append(best_epoch)
            test_MSEs.append(t_mse)

            # Save network weights to a file
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            #np.savez(save_path+'v3_sup.npz',
            #         *lasagne.layers.get_all_param_values(supervised_net))

        print("  learning rate:\t\t{:.9f}".format(float(lr.get_value())))
        print("  total time:\t\t\t{:.3f}s".format(time.time() - start_time))

        if epoch + 1 > best_epoch + 10:
            lr.set_value(lr.get_value() * 0.5)
            best_epoch = epoch + 1
            
        plot.clf()
        plot.plot(range(1, epoch+2), train_MSEs, label="train")
        plot.plot(range(1, epoch+2), valid_MSEs, label="valid")
        plot.plot(test_MSE_epochs, test_MSEs, 'ro', label="test")
        plot.axis([1, epoch+2, 0, 200])
        plot.legend()
        plot.savefig("training_curves.png")
        
    # Print summary of training
    best_epoch = test_MSE_epochs[-1]
    print("---------------")
    print("Best epoch : ", best_epoch)
    print("Training loss : ", train_MSEs[best_epoch-1])
    print("Validation loss : ", valid_MSEs[best_epoch-1])
    print("Test loss : ", test_MSEs[-1])
    
    


def main():
    parser = argparse.ArgumentParser(description="""Implementation of the
                                     feature selection v3""")
    parser.add_argument('dataset',
                        default='debug',
                        help='Dataset.')
    parser.add_argument('n_output',
                        default=100,
                        help='Output dimension.')
    parser.add_argument('--num_epochs',
                        '-ne',
                        type=int,
                        default=50,
                        help="""Optional. Int to indicate the max'
                        'number of epochs.""")

    parser.add_argument('--save_path',
                        '-sp',
                        type=str,
                        default='/data/lisatmp4/carriepl/FeatureSelection/',
                        help="""Optional. Save path for the model""")
    
    parser.add_argument('--kfold',
                        '-kf',
                        type=str,
                        default=None,
                        help='Optional. Params for k-fold validation. ' +
                             'Format : "no_fold/nb_folds"')

    args = parser.parse_args()
    
    if args.kfold is None:
        cross_validation = None
    else:
        no_folds = int(args.kfold.split("/")[0])
        nb_folds = int(args.kfold.split("/")[1])
        cross_validation = (no_folds, nb_folds)

    print ("Arguments: {}".format(args))
    execute(
        args.dataset,
        int(args.n_output),
        int(args.num_epochs),
        str(args.save_path),
        cross_validation)



if __name__ == '__main__':
    main()
