from sklearn.decomposition import RandomizedPCA, TruncatedSVD, PCA
from aggregate_dataset import load_data23andme_baselines
from feature_selection.experiments.common import dataset_utils, imdb
from feature_selection.experiments.variant2 import mainloop_helpers as mlh

import argparse
import os
import numpy as np
import time

def pca(dataset, n_comp_list, save_path, method="pca", which_fold=0):

    dataset_path = '/data/lisatmp4/romerosa/datasets/'
    embedding_source = '/data/lisatmp4/romerosa/datasets/1000_Genome_project/unsupervised_hist_3x26_fold'
    embedding_source += str(which_fold) + '.npy'
    print embedding_source
    # Load the dataset
    print("Loading data")
    x_train, y_train, x_valid, y_valid, x_test, y_test, \
        _, _ = mlh.load_data(
            dataset, dataset_path, embedding_source,
            which_fold=which_fold, keep_labels=1.0,
            missing_labels_val=-1.0,
            embedding_input='raw', norm=False)

    unsupervised = np.concatenate((x_train, x_valid), axis=0)

    save_path = os.path.join(save_path, dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if method == "pca":
        print "Applying PCA"
        # Extract PCA from unsupervised data
        pca = PCA()
        pca.fit(unsupervised)
        # Apply PCA to supervised training data
        new_x_train_supervised = pca.transform(x_train)
        new_x_valid_supervised = pca.transform(x_valid)
        new_x_test_supervised = pca.transform(x_test)

        # Remove items from n_comp_list that are outside the bounds
        max_n_comp = new_x_train_supervised.shape[1]
        n_comp_possible = [el for el in n_comp_list if el < max_n_comp]
        n_comp_list = n_comp_possible + [max_n_comp]

        print "Saving embeddings"
        for n_comp in n_comp_list:
            file_name = save_path + '/pca_' + str(n_comp) + '_embedding_fold' + str(which_fold) + '.npz'
            np.savez(file_name,
                     x_train_supervised=new_x_train_supervised[:, :n_comp],
                     y_train_supervised=y_train,
                     x_valid_supervised=new_x_valid_supervised[:, :n_comp],
                     y_valid_supervised=y_valid,
                     x_test_supervised=new_x_test_supervised[:, :n_comp],
                     y_test_supervised=y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""PCA embedding""")

    parser.add_argument('-d',
                        type=str,
                        default="1000_genomes",
                        help='dataset')
    parser.add_argument('-save_path',
                        '-sp',
                        default='/data/lisatmp4/romerosa/feature_selection/pca_final/',
                        help='number of components for embedding')

    args = parser.parse_args()

    for f in range(5):
        pca(args.d, [10, 50, 100],
            args.save_path, which_fold=f)
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
