from __future__ import print_function
import argparse
import time
import os

import lasagne
from lasagne.layers import DenseLayer, InputLayer
from lasagne.nonlinearities import sigmoid, softmax, tanh, linear
import numpy as np
import theano
import theano.tensor as T

#theano.config.compute_test_value = 'warn'


"""
NOTE : This script should be launched with the following Theano flags :
"device=gpu,floatX=float32,optimizer_excluding=scanOp_pushout_seqs_ops"
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


def group_softmax(z, group_size):

    g_z = z.reshape((-1, group_size))

    # The following line makes the softmax computation more numerically stable
    g_z = g_z - theano.gradient.zero_grad(g_z.max(axis=1, keepdims=True))

    # Compute the softmax along the group axis
    #g_act = T.nnet.softmax(g_z)
    exp = g_z.exp()
    g_act = exp / exp.sum(1)[:,None]

    act = g_act.reshape((z.shape[0], -1))
    return act


def group_categorical_crossentropy(y_hat, y, group_size):

    g_y_hat = y_hat.reshape((-1, group_size))
    g_y = y.reshape((-1, group_size))

    # Compute and return the loss
    target_classes = y.argmax(1)
    nll = -T.mean(T.log(y_hat)[T.arange(y.shape[0]), target_classes])

    return nll


# Main program
def execute(dataset, n_output, num_epochs=500):
    # Load the dataset
    print("Loading data")
    if dataset == 'genomics':
        from feature_selection.experiments.common.dorothea import load_data
        x_train, y_train = load_data('train', 'standard', False, 'numpy')
        x_valid, y_valid = load_data('valid', 'standard', False, 'numpy')

        # WARNING : The dorothea dataset has no test labels
        x_test = load_data('test', 'standard', False, 'numpy')
        y_test = None

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

        n_train = 10
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

        data = aggregate_dataset.load_data23andme(
                                        datapath='/data/lisatmp3/dejoieti',
                                        split=[.5, .25, .25], shuffle=False,
                                        seed=32)
        import pdb; pdb.set_trace()


    else:
        print("Unknown dataset")
        return

    n_samples, n_feats = x_train.shape
    input_cardinality = x_train.max() + 1
    n_classes = y_train.max() + 1
    n_batch = 10
    save_path = '/data/lisatmp4/carriepl/FeatureSelection/'

    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.ivector('targets')

    feature_var = theano.shared(x_train.transpose(), 'feature_var')
    lr = theano.shared(np.float32(1e-6), 'learning_rate')
    
    #input_var.tag.test_value = x_train[:20]
    #target_var.tag.test_value = y_train[:20]
    #feature_var.tag.test_value = x_train.transpose()

    # Build model
    print("Building model")

    # Define a few useful values for building the model
    feat_repr_size = 50
    h_rep_size = 50
    extended_input_size = input_cardinality * n_feats

    # Build the actual network input from the compressed input_var.
    inputs = [T.eq(input_var, float(i)).astype("float32")
              for i in range(input_cardinality)]
    inputs = T.stack(*inputs).dimshuffle(1, 2, 0).flatten(2)

    # Build the portion of the model that will predict the feature
    # weights from the feature activations
    def step(iteration_idx, dataset, inputs):
        # ALERT : It is necessary to launch this script with the proper Theano
        # flags or the following instructions will be pushed outside of the
        # Scan, increasing memory usage by a factor of roughly
        # 'input_cardinality'.
        activations = T.eq(dataset.astype("float32"),
                           iteration_idx.astype("float32")).astype("float32")

        feature_rep_net = InputLayer((n_feats, n_samples), activations)
        feature_rep_net = DenseLayer(feature_rep_net, num_units=feat_repr_size)
        feature_reps = lasagne.layers.get_output(feature_rep_net)

        # Predict the encoder parameters for the features
        enc_weights_net = DenseLayer(feature_rep_net, num_units=n_output)
        enc_weights_net = DenseLayer(enc_weights_net, num_units=h_rep_size,
                                W=lasagne.init.Uniform(), nonlinearity=tanh)
        encoder_weights = lasagne.layers.get_output(enc_weights_net)

        # Predict the contribution to the encoder's hidden representation
        # here and only return that because current memory constraints do not
        # allow us to return both 'encoder_weights' and 'decoder_weights' as
        # this would take too much memory.
        import pdb; pdb.set_trace()
        hidden_contribution = T.dot(inputs[:, iteration_idx::20],
                                    encoder_weights)

        # Predict the decoder parameters for the features
        dec_weights_net = DenseLayer(feature_rep_net, num_units=n_output)
        dec_weights_net = DenseLayer(dec_weights_net, num_units=h_rep_size,
                                 W=lasagne.init.Uniform(), nonlinearity=tanh)
        decoder_weights = lasagne.layers.get_output(dec_weights_net).transpose()

        return encoder_weights, decoder_weights

    results, updates = theano.scan(fn=step,
                                        sequences=T.arange(input_cardinality),
                                        non_sequences=[feature_var, inputs],
                                        allow_gc=False)

    encoder_weights = results[0].dimshuffle(1, 0, 2).reshape((-1, h_rep_size))
    decoder_weights = results[1].dimshuffle(1, 2, 0).reshape((h_rep_size, -1))

    # Build the encoder
    encoder = InputLayer((n_batch, extended_input_size), inputs)
    encoder = DenseLayer(encoder, num_units=n_output, W=encoder_weights)
    hidden_representation = lasagne.layers.get_output(encoder)

    # Build the decoder
    decoder = DenseLayer(encoder, num_units=extended_input_size,
                         W=decoder_weights, nonlinearity=linear)
    reconstruction = lasagne.layers.get_output(decoder)
    reconstruction = group_softmax(reconstruction, input_cardinality)

    # Build the supervised network that takes the hidden representation of the
    # encoder as input and tries to predict the targets
    supervised_net = DenseLayer(encoder, num_units=n_output)
    supervised_net = DenseLayer(encoder, num_units=1)
    prediction = lasagne.layers.get_output(supervised_net)[:, 0]


    # Compute loss expressions
    print("Defining loss functions")

    unsup_loss = group_categorical_crossentropy(reconstruction, inputs,
                                                input_cardinality)
    sup_loss = lasagne.objectives.squared_error(prediction, target_var).mean()
    total_loss = unsup_loss + sup_loss


    # Define training funtions
    print("Building training functions")

    params = (lasagne.layers.get_all_params(supervised_net) +
              lasagne.layers.get_all_params(decoder))

    # Do not train 'feature_var'. It's only a shared variable to avoid
    # transfers to/from the gpu
    params = [p for p in params if p is not feature_var]

    updates = lasagne.updates.rmsprop(total_loss,
                                      params,
                                      learning_rate=lr)
    # updates = lasagne.updates.sgd(total_loss,
    #                              params,
    #                              learning_rate=lr)
    # updates = lasagne.updates.momentum(total_loss, params,
    #                                    learning_rate=lr, momentum=0.0)
    updates[lr] = (lr * 0.99).astype("float32")

    train_fn = theano.function([input_var, target_var], total_loss,
                               updates=updates)

    #theano.printing.pydotprint(train_fn, "fct.png", scan_graphs=True)

    # Define monitoring functions
    print("Building monitoring functions")

    test_reconstruction = lasagne.layers.get_output(decoder,
                                                    deterministic=True)
    test_reconstruction = group_softmax(test_reconstruction, input_cardinality)

    test_unsup_loss = group_categorical_crossentropy(test_reconstruction,
                                                     inputs, input_cardinality)

    test_prediction = lasagne.layers.get_output(supervised_net,
                                                deterministic=True)
    test_sup_loss = lasagne.objectives.squared_error(prediction,
                                                     target_var).mean()

    val_fn = theano.function([input_var, target_var],
                             [test_unsup_loss, test_sup_loss])
    monitor_labels = ["reconstruction loss", "prediction loss"]


    # Finally, launch the training loop.
    print("Starting training...")

    # We iterate over epochs:
    best_valid_mse = 1e20
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
        monitoring(train_minibatches, "train", val_fn, monitor_labels)

        # Monitor progress on the validation set
        valid_minibatches = iterate_minibatches(x_valid, y_valid, n_batch,
                                                shuffle=False)
        mse = monitoring(valid_minibatches, "valid",
                         val_fn, monitor_labels)['prediction loss']

        # Monitor the test set if needed
        if mse < best_valid_mse:
            best_valid_mse = mse

            test_minibatches = iterate_minibatches(x_test, y_test, n_batch,
                                                   shuffle=False)
            monitoring(test_minibatches, "test", val_fn, monitor_labels)

            # Save network weights to a file
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            np.savez(save_path+'v3_unsup.npz',
                     *lasagne.layers.get_all_param_values(decoder))
            np.savez(save_path+'v3_sup.npz',
                     *lasagne.layers.get_all_param_values(supervised_net))

        print("  total time:\t\t\t{:.3f}s".format(time.time() - start_time))


def main():
    parser = argparse.ArgumentParser(description="""Implementation of the
                                     feature selection v2""")
    parser.add_argument('dataset',
                        default='debug',
                        help='Dataset.')
    parser.add_argument('n_output',
                        default=100,
                        help='Output dimension.')
    parser.add_argument('--num_epochs',
                        '-ne',
                        type=int,
                        default=5,
                        help="""Optional. Int to indicate the max'
                        'number of epochs.""")

    args = parser.parse_args()

    execute(args.dataset, int(args.n_output), int(args.num_epochs))



if __name__ == '__main__':
    main()
