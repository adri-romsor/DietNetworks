from __future__ import print_function
import os
import numpy as np

import lasagne
from lasagne.layers import DenseLayer, InputLayer, DropoutLayer, BatchNormLayer
from lasagne.nonlinearities import (sigmoid, softmax, tanh, linear, rectify,
                                    leaky_rectify, very_leaky_rectify)
from lasagne.regularization import apply_penalty, l2, l1
from lasagne.init import Uniform
import theano
import theano.tensor as T

_EPSILON = 10e-8


def build_feat_emb_nets(embedding_source, n_feats, n_samples_unsup,
                        input_var_unsup, n_hidden_u, n_hidden_t_enc,
                        n_hidden_t_dec, gamma, encoder_net_init,
                        decoder_net_init, save_path):

    nets = []
    embeddings = []

    if not embedding_source:  # meaning we haven't done any unsup pre-training
        encoder_net = InputLayer((n_feats, n_samples_unsup), input_var_unsup)
        for i, out in enumerate(n_hidden_u):
            encoder_net = DenseLayer(encoder_net, num_units=out,
                                     nonlinearity=rectify)
            # encoder_net = DropoutLayer(encoder_net)
        feat_emb = lasagne.layers.get_output(encoder_net)
        pred_feat_emb = theano.function([], feat_emb)
    else:  # meaning we haven done some unsup pre-training
        feat_emb_val = np.load(os.path.join(save_path.rsplit('/', 1)[0],
                                            embedding_source)).items()[0][1]
        feat_emb = theano.shared(feat_emb_val, 'feat_emb')
        encoder_net = InputLayer((n_feats, n_hidden_u[-1]), feat_emb)

    # Build transformations (f_theta, f_theta') network and supervised network
    # f_theta (ou W_enc)
    encoder_net_W_enc = encoder_net
    for hid in n_hidden_t_enc:
        encoder_net_W_enc = DenseLayer(encoder_net_W_enc, num_units=hid,
                                       nonlinearity=tanh,  # tanh
                                       W=Uniform(encoder_net_init)
                                       )
        # encoder_net_W_enc = DropoutLayer(encoder_net_W_enc)
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
            units = [n_feats] + n_hidden_u + n_hidden_t[i][:-1]
            units.reverse()
            W_net = enc_nets[i]
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
                      n_hidden_s, embedding, disc_nonlinearity, n_targets):
    # Supervised network
    discrim_net = InputLayer((batch_size, n_feats), input_var_sup)
    discrim_net = DenseLayer(discrim_net, num_units=n_hidden_t_enc[-1],
                             W=embedding, nonlinearity=rectify)

    # Supervised hidden layers
    for hid in n_hidden_s:
        discrim_net = DropoutLayer(discrim_net)
        discrim_net = DenseLayer(discrim_net, num_units=hid)

    # Predicting labels
    assert disc_nonlinearity in ["sigmoid", "linear", "rectify", "softmax"]
    discrim_net = DropoutLayer(discrim_net)
    discrim_net = DenseLayer(discrim_net, num_units=n_targets,
                             nonlinearity=eval(disc_nonlinearity))

    return discrim_net


def build_reconst_net(discrim_net, embedding, n_feats, gamma):
    # Reconstruct the input using dec_feat_emb
    if gamma > 0:
        lays = lasagne.layers.get_all_layers(discrim_net)
        reconst_net = lays[-3]

        reconst_net = DenseLayer(reconst_net, num_units=n_feats,
                                 W=embedding.T)
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
    elif disc_nonlinearity == "softmax":
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


def definte_test_functions(disc_nonlinearity, prediction, prediction_det,
                           target_var_sup):
    if disc_nonlinearity in ["sigmoid", "softmax"]:
        if disc_nonlinearity == "sigmoid":
            test_pred = T.gt(prediction_det, 0.5)
            test_acc = T.mean(T.eq(test_pred, target_var_sup),
                              dtype=theano.config.floatX) * 100.

        elif disc_nonlinearity == "softmax":
            test_pred = prediction_det.argmax(1)
            test_acc = T.mean(T.eq(test_pred, target_var_sup.argmax(1)),
                              dtype=theano.config.floatX) * 100
        return test_acc, test_pred
