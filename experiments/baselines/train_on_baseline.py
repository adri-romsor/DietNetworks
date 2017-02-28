#!/usr/bin/env python2

import os

from featsel_supervised import execute


def main(embedding_path, n_classes, which_method="pca", enc='triangle'):

    print "Starting main loop"
    embedding_methods = []

    print "Method: {}".format(which_method)

    embedding_methods = os.listdir(embedding_path)
    embedding_methods = [emb for emb in embedding_methods
                         if ".npz" in emb and which_method in emb]
    if which_method == "kmeans":
        embedding_methods = [emb for emb in embedding_methods if
                             enc in emb]

    print "Embedding methods: {}".format(embedding_methods)
    mod = 1
    # lr_candidates = [1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    lr_candidates = [1e-3]
    for lr_value in lr_candidates:
        for embedding in embedding_methods:
            print "Training model %s: lr %d out of %d" % \
                    (embedding, mod, len(lr_candidates))
            execute(embedding, num_epochs=500, lr_value=lr_value,
                    n_classes=n_classes, save_path=embedding_path)
        mod += 1

if __name__ == '__main__':

    embedding_path = "/data/lisatmp4/romerosa/DietNetworks/pca_final/"
    dataset = '1000_genomes'
    n_classes = 26
    which_method = 'pca'
    embedding_path = os.path.join(embedding_path, dataset)

    main(embedding_path, n_classes, which_method, enc="hard")
