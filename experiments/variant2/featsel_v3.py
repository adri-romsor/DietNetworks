from __future__ import print_function
import argparse
import time
import os

import lasagne
from lasagne.layers import DenseLayer, InputLayer
from lasagne.nonlinearities import sigmoid, softmax, tanh, linear, rectify
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


def weights(n_in, n_out, name=None):
    return theano.shared(np.random.uniform(-0.01, 0.01,
                                           size=(n_in, n_out)).astype("float32"),
                         name)
    
    
def biases(n_out, name=None):
    return theano.shared(np.zeros((n_out,), "float32"), name)

    
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

        data = aggregate_dataset.load_data23andme(
                                        data_path='/data/lisatmp4/dejoieti',
                                        split=[.5, .75, 1.0], shuffle=False,
                                        seed=32)
        
        tmp = []
        total0 = [0] * 12
        total1 = [0] * 12
        sets = [set() for i in range(12)]
        acceptable_quadriplets = [(0, 2, 3, 7),    # AA-AC-CC
                                  (0, 2, 4, 14),   # AA-AG-GG
                                  (0, 2, 5, 19),   # AA-AT-TT
                                  (0, 7, 8, 14),   # CC-CG-GG
                                  (0, 7, 9, 19),   # CC-CT-TT
                                  (0, 14, 15, 19), # GG-GT-TT
                                  (0, 11, 12, 17)] # DD-DI-II
        feature_w_quad = [0, 0]
        
        for i in range(data[0].shape[1]):
            if i % 1000 == 0:
                print(i, data[0].shape[1])
                
            uniques = np.unique(data[0][:,i])
            nb_uniques = len(uniques)
                
            tmp.append(nb_uniques)

            if uniques[0] == 0:
                total0[nb_uniques] += 1
            else:
                total1[nb_uniques] += 1
                
            mytuple = tuple([i for i in uniques])
            if mytuple not in sets[nb_uniques]:
                sets[nb_uniques].add(mytuple)
                
            if nb_uniques == 4:
                if mytuple in acceptable_quadriplets:
                    feature_w_quad[0] += 1
                else:
                    feature_w_quad[1] += 1
                
            
        import pdb; pdb.set_trace()
        
        # Trim the features of the dataset
        features_to_keep = []
        for i in range(data[0].shape[1]):
            if i % 1000 == 0:
                print(i, data[0].shape[1])
                
            uniques = np.unique(data[0][:,i])
            nb_uniques = len(uniques)
            
            if nb_uniques > 2:
                features_to_keep.append(i)
        
        
        
        x_train = data[4].astype("int8")[:10,features_to_keep][:,::5]
        x_valid = data[5].astype("int8")[:10,features_to_keep][:,::5]
        x_test = data[6].astype("int8")[:10,features_to_keep][:,::5]
        
        y_train = data[7].astype("float32")[:10]
        y_valid = data[8].astype("float32")[:10]
        y_test = data[9].astype("float32")[:10]
        
        
        print(x_train.shape, x_valid.shape, x_test.shape)
        
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
    input_var = T.fmatrix('inputs')
    target_var = T.fvector('targets')
    
    feature_var = theano.shared(x_train.transpose().astype("float32"), 'feature_var')
    lr = theano.shared(np.float32(3e-4), 'learning_rate')
    
    #input_var.tag.test_value = x_train[:20]
    #target_var.tag.test_value = y_train[:20]
    #feature_var.tag.test_value = x_train.transpose()

    # Build model
    print("Building model")
    
    # Define a few useful values for building the model
    feat_repr_size = 20
    h_rep_size = 20
    extended_input_size = input_cardinality * n_feats
    
    # Define the network architecture             
    params = {
        'feature': {
            'layers': [feat_repr_size],
            'weights': [weights(n_samples, feat_repr_size, 'fw1')],
            'biases': [biases(feat_repr_size, 'fb1')],
            'acts': [rectify]},
        'encoder_w': {
            'layers': [n_output, h_rep_size],
            'weights': [weights(feat_repr_size, n_output, 'ew1'),
                        weights(n_output, h_rep_size, 'ew2')],
            'biases': [biases(n_output, 'eb1'),
                       biases(h_rep_size, 'eb2')],
            'acts': [rectify, tanh]},
        'decoder_w': {
            'layers': [n_output, h_rep_size],
            'weights': [weights(feat_repr_size, n_output, 'dw1'),
                        weights(n_output, h_rep_size, 'dw2')],
            'biases': [biases(n_output, 'db1'),
                       biases(h_rep_size, 'db2')],
            'acts': [rectify, tanh]},
        'supervised': {
            'layers': [n_output, 1],
            'weights': [weights(h_rep_size, n_output, 'sw1'),
                        weights(n_output, 1, 'sw2')],
            'biases': [biases(n_output, 'sb1'),
                       biases(1, 'sb2')],
            'acts': [rectify, linear]}}
    
    # Build the actual network input from the compressed input_var.
    inputs = [T.eq(input_var, float(i)).astype("float32")
              for i in range(input_cardinality)]
    inputs = T.stack(*inputs).dimshuffle(1, 2, 0).flatten(2)
    
    # Build the portion of the model that will predict the encoder's hidden
    # representation fron the inputs and the feature activations.
    def step_enc(iteration_idx, dataset, inputs):
        # ALERT : It is necessary to launch this script with the proper Theano
        # flags or the following instructions will be pushed outside of the
        # Scan, increasing memory usage by a factor of roughly
        # 'input_cardinality'.
        activations = T.eq(dataset.astype("float32"),
                           iteration_idx.astype("float32")).astype("float32")

        # Predict the feature representations 
        feature_rep_net = InputLayer((n_feats, n_samples), activations)
        for i in range(len(params['feature']['layers'])):
            feature_rep_net = DenseLayer(feature_rep_net,
                                num_units=params['feature']['layers'][i],
                                W=params['feature']['weights'][i],
                                b=params['feature']['biases'][i],
                                nonlinearity=params['feature']['acts'][i])
        
        # Predict the encoder parameters for the features
        enc_weights_net = feature_rep_net
        for i in range(len(params['encoder_w']['layers'])):
            enc_weights_net = DenseLayer(enc_weights_net,
                                num_units=params['encoder_w']['layers'][i],
                                W=params['encoder_w']['weights'][i],
                                b=params['encoder_w']['biases'][i],
                                nonlinearity=params['encoder_w']['acts'][i])
        encoder_weights = lasagne.layers.get_output(enc_weights_net)
        
        # Predict the contribution to the encoder's hidden representation.
        hidden_contribution = T.dot(inputs[:, iteration_idx::20],
                                    encoder_weights)
        
        return hidden_contribution
        
    results, updates = theano.scan(fn=step_enc,
                                   sequences=T.arange(input_cardinality),
                                   non_sequences=[feature_var, inputs],
                                   allow_gc=True, name="encoder_w_scan")
    hidden_contribution = results.sum(0)
    
    # Add a bias vector to the hidden_contribution and apply an activation
    # function to obtain the autoencoder's hidden representation
    encoder_b = biases(h_rep_size, 'encoder_b')
    h_rep = T.nnet.relu(hidden_contribution + encoder_b)
    
    # Build the decoder
    def step_dec(iteration_idx, dataset, h_rep):
        # Compute the feature representations
        activations = T.eq(dataset.astype("float32"),
                           iteration_idx.astype("float32")).astype("float32")
        
        # Predict the feature representations 
        feature_rep_net = InputLayer((n_feats, n_samples), activations)
        for i in range(len(params['feature']['layers'])):
            feature_rep_net = DenseLayer(feature_rep_net,
                                num_units=params['feature']['layers'][i],
                                W=params['feature']['weights'][i],
                                b=params['feature']['biases'][i],
                                nonlinearity=params['feature']['acts'][i])
        
        # Predict the decoder parameters for the features
        dec_weights_net = feature_rep_net
        for i in range(len(params['decoder_w']['layers'])):
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
                                   sequences=T.arange(input_cardinality),
                                   non_sequences=[feature_var, h_rep],
                                   allow_gc=True, name="decoder_w_scan")
    pre_act_reconstruction = results.dimshuffle(1, 2, 0).flatten(2)
    reconstruction = group_softmax(pre_act_reconstruction, input_cardinality)
    
    # Build the supervised network that takes the hidden representation of the
    # encoder as input and tries to predict the targets
    supervised_net = InputLayer((n_batch, h_rep_size), h_rep)
    for i in range(len(params['supervised']['layers'])):
        supervised_net = DenseLayer(supervised_net,
                                num_units=params['supervised']['layers'][i],
                                W=params['supervised']['weights'][i],
                                b=params['supervised']['biases'][i],
                                nonlinearity=params['supervised']['acts'][i])
    prediction = lasagne.layers.get_output(supervised_net)[:, 0]

    # Compute loss expressions
    print("Defining loss functions")    
    
    unsup_loss = group_categorical_crossentropy(reconstruction, inputs,
                                                input_cardinality)
    
    
    prediction_mse = (target_var - prediction) ** 2
    sup_loss = (prediction_mse * T.neq(target_var, -1)).mean()

    total_loss = unsup_loss + sup_loss
    
    
    # Define training funtions
    print("Building training functions")
    
    # There are 3 places where the parameters are definer:
    # - In the 'params' dictionnary
    # - The encoder biases
    # - The params in the 'supervised_net' network
    params = (lasagne.layers.get_all_params(supervised_net) +
              [encoder_b] +
              params['feature']['weights'] + 
              params['feature']['biases'] + 
              params['encoder_w']['weights'] + 
              params['encoder_w']['biases'] + 
              params['decoder_w']['weights'] + 
              params['decoder_w']['biases'])
    
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
    val_fn = theano.function([input_var, target_var], [unsup_loss, sup_loss])
    monitor_labels = ["reconstruction loss", "prediction loss"]
    
    #theano.printing.pydotprint(train_fn, "fct.png", scan_graphs=True)   
    
    
    # Define monitoring functions
    print("Building monitoring functions")
    
    val_fn = theano.function([input_var, target_var], [unsup_loss, sup_loss])
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
        
        print(time.time() - start_time)

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

            #np.savez(save_path+'v3_sup.npz',
            #         *lasagne.layers.get_all_param_values(supervised_net))
                
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
