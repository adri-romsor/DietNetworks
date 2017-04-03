from __future__ import print_function
import os
import numpy as np

import lasagne
from lasagne.layers import DenseLayer, InputLayer, DropoutLayer, BatchNormLayer, \
    MergeLayer, Layer
from lasagne.nonlinearities import (sigmoid, softmax, tanh, linear, rectify,
                                    leaky_rectify, very_leaky_rectify)
from lasagne.regularization import apply_penalty, l2, l1
from lasagne.init import Uniform
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

_EPSILON = 10e-8


def build_feat_emb_nets(embedding_source, n_feats, n_samples_unsup,
                        input_var_unsup, n_hidden_u, n_hidden_t_enc,
                        n_hidden_t_dec, gamma, encoder_net_init,
                        decoder_net_init, save_path, random_proj=False):

    nets = []
    embeddings = []

    if not embedding_source:  # meaning we haven't done any unsup pre-training
        encoder_net = InputLayer((n_feats, n_samples_unsup), input_var_unsup)
        for i, out in enumerate(n_hidden_u):
            encoder_net = DenseLayer(encoder_net, num_units=out,
                                     nonlinearity=rectify)
            if random_proj:
                freezeParameters(encoder_net)
            # encoder_net = DropoutLayer(encoder_net)
            # encoder_net = BatchNormLayer(encoder_net)
        feat_emb = lasagne.layers.get_output(encoder_net)
        pred_feat_emb = theano.function([], feat_emb)
    else:  # meaning we have done some unsup pre-training
        if os.path.exists(embedding_source):  # embedding_source is a path itself
            path_to_load = embedding_source
        else:  # fetch the embedding_source file in save_path
            path_to_load = os.path.join(save_path.rsplit('/', 1)[0],
                                        embedding_source)
        if embedding_source[-3:] == "npz":
            feat_emb_val = np.load(path_to_load).items()[0][1]
        else:
            feat_emb_val = np.load(path_to_load)
            feat_emb_val = feat_emb_val.astype('float32')
            # means = feat_emb_val.mean(1)
            # stds = feat_emb_val.std(1)
            # feat_emb_val  = feat_emb_val-means[:, None]
            # feat_emb_val /= stds[:, None]


        feat_emb = theano.shared(feat_emb_val, 'feat_emb')
        encoder_net = InputLayer((n_feats, feat_emb_val.shape[1]), feat_emb)

    # Build transformations (f_theta, f_theta') network and supervised network
    # f_theta (ou W_enc)
    encoder_net_W_enc = encoder_net
    for hid in n_hidden_t_enc:
        encoder_net_W_enc = DenseLayer(encoder_net_W_enc, num_units=hid,
                                       nonlinearity=tanh,  # tanh
                                       W=Uniform(encoder_net_init)
                                       )
        # encoder_net_W_enc = DropoutLayer(encoder_net_W_enc)
        # encoder_net = BatchNormLayer(encoder_net_W_enc)
    enc_feat_emb = lasagne.layers.get_output(encoder_net_W_enc)

    nets.append(encoder_net_W_enc)
    embeddings.append(enc_feat_emb)

    # f_theta' (ou W_dec)
    if gamma > 0:  # meaning we are going to train to reconst the fat data
        encoder_net_W_dec = encoder_net
        for hid in n_hidden_t_dec:
            encoder_net_W_dec = DenseLayer(encoder_net_W_dec, num_units=hid,
                                           nonlinearity=tanh,  # tanh
                                           W=Uniform(decoder_net_init)
                                           )
            # encoder_net_W_dec = DropoutLayer(encoder_net_W_dec)
            # encoder_net = BatchNormLayer(encoder_net_W_dec)
        dec_feat_emb = lasagne.layers.get_output(encoder_net_W_dec)

    else:
        encoder_net_W_dec = None
        dec_feat_emb = None

    nets.append(encoder_net_W_dec)
    embeddings.append(dec_feat_emb)

    return [nets, embeddings, pred_feat_emb if not embedding_source else []]


def build_feat_emb_reconst_nets(coeffs, n_feats, n_hidden_u,
                                n_hidden_t, enc_nets, net_inits):

    nets = []

    for i, c in enumerate(coeffs):
        if c > 0:
            units = [n_feats] + n_hidden_u[:-1]
            units.reverse()
            W_net = enc_nets[i]
            lays = lasagne.layers.get_all_layers(W_net)
            lays_dense = [el for el in lays if isinstance(el, DenseLayer)]
            W_net = lays_dense[len(n_hidden_u)-1]
            for u in units:
                # Add reconstruction of the feature embedding
                W_net = DenseLayer(W_net, num_units=u,
                                   nonlinearity=linear,
                                   W=Uniform(net_inits[i]))
                # W_net = DropoutLayer(W_net)
        else:
            W_net = None
        nets += [W_net]

    return nets


def build_discrim_net(batch_size, n_feats, input_var_sup, n_hidden_t_enc,
                      n_hidden_s, embedding, disc_nonlinearity, n_targets,
                      batchnorm=False):
    # Supervised network
    discrim_net = InputLayer((batch_size, n_feats), input_var_sup)
    discrim_net = DenseLayer(discrim_net, num_units=n_hidden_t_enc[-1],
                             W=embedding, nonlinearity=rectify)
    hidden_rep = discrim_net

    # Supervised hidden layers
    for hid in n_hidden_s:
        if batchnorm:
            discrim_net = BatchNormLayer(discrim_net)
        discrim_net = DropoutLayer(discrim_net)
        # discrim_net = BatchNormLayer(discrim_net)
        discrim_net = DenseLayer(discrim_net, num_units=hid)

    # Predicting labels
    assert disc_nonlinearity in ["sigmoid", "linear", "rectify",
                                 "softmax", "softmax_hierarchy"]
    if batchnorm:
        discrim_net = BatchNormLayer(discrim_net)
    discrim_net = DropoutLayer(discrim_net)
    if disc_nonlinearity != "softmax_hierarchy":
        discrim_net = DenseLayer(discrim_net, num_units=n_targets,
                                 nonlinearity=eval(disc_nonlinearity))
    else:
        cont_labels = create_1000_genomes_continent_labels()
        hierarch_softmax_1000_genomes = HierarchicalSoftmax(cont_labels)
        discrim_net_e= DenseLayer(discrim_net, num_units=n_targets,
                                  nonlinearity=hierarch_softmax_1000_genomes)
        discrim_net_c= DenseLayer(discrim_net, num_units=len(cont_labels),
                                  nonlinearity=softmax)
        discrim_net = HierarchicalMergeSoftmaxLayer([discrim_net_e,
                                                     discrim_net_c],
                                                     cont_labels)
    return discrim_net, hidden_rep


def build_reconst_net(hidden_rep, embedding, n_feats, gamma):
    # Reconstruct the input using dec_feat_emb
    if gamma > 0:
        reconst_net = DenseLayer(hidden_rep, num_units=n_feats,
                                 W=embedding.T, nonlinearity=rectify)
    else:
        reconst_net = None

    return reconst_net


def define_predictions(nets, start=0):
    preds = []
    preds_det = []

    for i, n in enumerate(nets):
        if i >= start and n is None:
            preds += [None]
            preds_det += [None]
        elif i >= start and n is not None:
            preds += [lasagne.layers.get_output(nets[i])]
            preds_det += [lasagne.layers.get_output(nets[i],
                                                    deterministic=True)]

    return preds, preds_det


def define_reconst_losses(preds, preds_det, input_vars_list):
    reconst_losses = []
    reconst_losses_det = []

    for i, p in enumerate(preds):
        if p is None:
            reconst_losses += [0]
            reconst_losses_det += [0]
        else:
            reconst_losses += [lasagne.objectives.squared_error(
                p, input_vars_list[i]).mean()]
            reconst_losses_det += [lasagne.objectives.squared_error(
                preds_det[i], input_vars_list[i]).mean()]

    return reconst_losses, reconst_losses_det


def define_loss(pred, pred_det, target_var, output_type):

    if output_type == 'raw' or output_type == 'w2v':  # loss is MSE
        loss = lasagne.objectives.squared_error(pred, target_var).mean()
        loss_det = \
            lasagne.objectives.squared_error(pred_det, target_var).mean()
    elif 'histo' in output_type:  # loss is crossentropy
        loss = crossentropy(pred, target_var).mean()
        loss_det = crossentropy(pred_det, target_var).mean()
    elif output_type == 'bin':  # loss is binary_crossentropy
        loss = lasagne.objectives.binary_crossentropy(pred, target_var).mean()
        loss_det = \
            lasagne.objectives.binary_crossentropy(pred_det, target_var).mean()

    return loss, loss_det


def crossentropy(y_pred, y_true):
    # Clip probs
    y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
    y_true = T.clip(y_true, _EPSILON, 1.0 - _EPSILON)

    # Compute cross-entropy
    loss = T.nnet.categorical_crossentropy(y_pred, y_true)

    return loss


def define_sup_loss(disc_nonlinearity, prediction, prediction_det, keep_labels,
                    target_var_sup, missing_labels_val):
    if disc_nonlinearity == "sigmoid":
        clipped_prediction = T.clip(prediction, _EPSILON, 1.0 - _EPSILON)
        clipped_prediction_det = T.clip(prediction_det, _EPSILON,
                                        1.0 - _EPSILON)
        loss_sup = lasagne.objectives.binary_crossentropy(
            clipped_prediction, target_var_sup)
        loss_sup_det = lasagne.objectives.binary_crossentropy(
            clipped_prediction_det, target_var_sup)
    elif disc_nonlinearity in ["softmax", "softmax_hierarchy"]:
        clipped_prediction = T.clip(prediction, _EPSILON, 1.0 - _EPSILON)
        clipped_prediction_det = T.clip(prediction_det, _EPSILON,
                                        1.0 - _EPSILON)
        loss_sup = lasagne.objectives.categorical_crossentropy(
            clipped_prediction,
            target_var_sup)
        loss_sup_det = lasagne.objectives.categorical_crossentropy(
            clipped_prediction_det,
            target_var_sup)
    elif disc_nonlinearity in ["linear", "rectify"]:
        loss_sup = lasagne.objectives.squared_error(
            prediction, target_var_sup)
        loss_sup_det = lasagne.objectives.squared_error(
            prediction_det, target_var_sup)
    else:
        raise ValueError("Unsupported non-linearity")

    # If some labels are missing, mask the appropriate losses before taking
    # the mean.
    if keep_labels < 1.0:
        mask = T.neq(target_var_sup, missing_labels_val)
        scale_factor = 1.0 / mask.mean()
        loss_sup = (loss_sup * mask) * scale_factor
        loss_sup_det = (loss_sup_det * mask) * scale_factor
    loss_sup = loss_sup.mean()
    loss_sup_det = loss_sup_det.mean()

    return loss_sup, loss_sup_det


def define_test_functions(disc_nonlinearity, prediction, prediction_det,
                           target_var_sup):
    if disc_nonlinearity in ["sigmoid", "softmax", "softmax_hierarchy"]:
        if disc_nonlinearity == "sigmoid":
            test_pred = T.gt(prediction_det, 0.5)
            test_acc = T.mean(T.eq(test_pred, target_var_sup),
                              dtype=theano.config.floatX) * 100.

        elif disc_nonlinearity in ["softmax", "softmax_hierarchy"]:
            test_pred = prediction_det.argmax(1)
            test_acc = T.mean(T.eq(test_pred, target_var_sup.argmax(1)),
                              dtype=theano.config.floatX) * 100
        return test_acc, test_pred


def create_1000_genomes_continent_labels():
    labels = ['ACB', 'ASW', 'BEB', 'CDX', 'CEU', 'CHB', 'CHS', 'CLM', 'ESN',
              'FIN', 'GBR', 'GIH', 'GWD', 'IBS', 'ITU', 'JPT', 'KHV', 'LWK',
               'MSL', 'MXL', 'PEL', 'PJL', 'PUR', 'STU', 'TSI', 'YRI']

    eas = []
    eur = []
    afr = []
    amr = []
    sas = []

    for i, l in enumerate(labels):
        if l in ['CHB', 'JPT', 'CHS', 'CDX', 'KHV']:
            eas +=  [i]  # EAS
        elif l in ['CEU', 'TSI', 'FIN', 'GBR', 'IBS']:
            eur += [i]  # EUR
        elif l in ['YRI', 'LWK', 'GWD', 'MSL', 'ESN', 'ASW', 'ACB']:
            afr += [i]  # AFR
        elif l in ['MXL', 'PUR', 'CLM', 'PEL']:
            amr += [i]  # AMR
        elif l in ['GIH', 'PJL', 'BEB', 'STU', 'ITU']:
            sas += [i]  # SAS

    cont_labels  = [eas, eur, afr, amr, sas]

    return cont_labels

class HierarchicalSoftmax(object):
    """
    Parameters
    ----------
    group_labels : list of lists of ints
        List of lists, where each sublist represents a group.
        Each sublist contains the indices belonging to the same subgroup.
    Methods
    -------
    __call__(x)
        Apply the hierarchical softmax function to the activation `x`.
    """
    def __init__(self, group_labels=[[1,2], [3,4]]):
        self.group_labels = group_labels

    def __call__(self, x):
        softmax_hierarchy = T.zeros_like(x)

        # Compute softmax output per group
        for g in range(len(self.group_labels)):
            mask = T.zeros_like(x,  dtype='float32')
            for el in self.group_labels[g]:
                mask = T.set_subtensor(mask[:, el], 1.0)

            cont_x = x*mask
            stable_cont_x = cont_x - \
                theano.gradient.zero_grad(cont_x.max(1, keepdims=True))
            exp_cont_x = stable_cont_x.exp() * mask
            softmax_cont_x = exp_cont_x / exp_cont_x.sum(1)[:, None]

            softmax_hierarchy = softmax_hierarchy + softmax_cont_x

        return softmax_hierarchy


class HierarchicalMergeSoftmaxLayer(MergeLayer):
    """
    This layer performs an elementwise merge of its input layers.
    It requires all input layers to have the same output shape.
    Parameters
    ----------
    incomings : a list of :class:`Layer` instances or tuples
        the layers feeding into this layer
    group_labels : a lit of lists, where each sublist contains the labels of
        each group.
    """

    def __init__(self, incomings, group_labels, **kwargs):
        # self.input_shapes = [incomings[0].output_shape,
        #                      incomings[1].output_shape]
        # self.input_layers = [None if isinstance(incoming, tuple)
        #                      else incoming
        #                      for incoming in incomings]
        self.group_labels  = group_labels

        self.get_output_kwargs = []
        super(HierarchicalMergeSoftmaxLayer, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        output_shape = self.input_shapes[0]

        return output_shape

    def get_output_for(self, inputs, **kwargs):

        # Combine 2 softmax layers
        mask = T.zeros_like(inputs[0],  dtype='float32')
        for c in range(len(self.group_labels)):
            for el in self.group_labels[c]:
                mask = T.set_subtensor(mask[:, el], inputs[1][:, c])

        output = inputs[0]*mask

        return output


def define_sampled_mean_bincrossentropy(y_pred, x, gamma=.5, one_ratio=.25,
                                        random_stream=RandomStreams(seed=1)):

    noisy_x = x + random_stream.binomial(size=x.shape, n=1,
                                         prob=one_ratio, ndim=None)
    p = T.switch(noisy_x > 0, 1, 0)
    p = T.cast(p, theano.config.floatX)

    # L1 penalty on activations
    l1_penalty = T.abs_(y_pred).mean()

    y_pred_p = T.clip(y_pred*p, _EPSILON, 1.0 - _EPSILON)
    x = T.clip(x, _EPSILON, 1.0 - _EPSILON)

    cost = lasagne.objectives.binary_crossentropy(y_pred_p, x)

    cost = (cost * p).mean()

    cost = cost + gamma*l1_penalty

    return cost


def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = T.flatten(y_true)
    y_pred_f = T.flatten(y_pred)
    intersection = T.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (T.sum(y_true_f) + T.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def freezeParameters(net, single=True):
    all_layers = lasagne.layers.get_all_layers(net)

    if single:
        all_layers = [all_layers[-1]]

    for layer in all_layers:
        layer_params = layer.get_params()
        for p in layer_params:
            try:
                layer.params[p].remove('trainable')
            except KeyError:
                pass


def rectify_minus2(x):
    """Rectify activation function :math:`\\varphi(x) = \\max(0, x)`
    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).
    Returns
    -------
    float32
        The output of the rectify function applied to the activation.
    """
    return theano.tensor.nnet.relu(x+2)-2
