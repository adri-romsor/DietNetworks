from sklearn.decomposition import RandomizedPCA, TruncatedSVD, PCA
from aggregate_dataset import load_data23andme_baselines
from feature_selection.experiments.common import dataset_utils, imdb

import argparse
import os
import numpy as np
import time

def pca(dataset, n_comp_list, save_path, method="pca"):

    # Load the dataset
    print("Loading data")
    splits = [0.6, 0.2]  # This will split the data into [60%, 20%, 20%]

    if dataset == "opensnp":
        x_train, x_test, unsupervised = load_data23andme_baselines()
    elif dataset == '1000_genomes':
        data = dataset_utils.load_1000_genomes(transpose=False,
                                               label_splits=splits)
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test),\
            x_nolabel = data
        unsupervised = np.concatenate((x_train, x_valid), axis=0)
    else:
        raise ValueError("Unknown dataset")

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
        new_x_test_supervised = pca.transform(x_test)

        if dataset != 'opensnp':
            new_x_valid_supervised = pca.transform(x_valid)
        else:
            new_x_valid_supervised = None

        # Remove items from n_comp_list that are outside the bounds
        max_n_comp = new_x_train_supervised.shape[1]
        n_comp_possible = [el for el in n_comp_list if el < max_n_comp]
        n_comp_list = n_comp_possible + [max_n_comp]

        print "Saving embeddings"
        for n_comp in n_comp_list:
            file_name = save_path + '/pca_' + str(n_comp) + '_embedding.npz'
            np.savez(file_name,
                     x_train_supervised=new_x_train_supervised[:, :n_comp],
                     y_train_supervised=y_train,
                     x_valid_supervised=new_x_valid_supervised[:, :n_comp],
                     y_valid_supervised=y_valid,
                     x_test_supervised=new_x_test_supervised[:, :n_comp],
                     y_test_supervised=y_test)

    elif method == "randPCA":
        print "Applying PCA"
        # Extract PCA from unsupervised data
        pca = RandomizedPCA()
        pca.fit(unsupervised)
        # Apply PCA to supervised training data
        new_x_train_supervised = pca.transform(x_train)
        new_x_test_supervised = pca.transform(x_test)

        if dataset != 'opensnp':
            new_x_valid_supervised = pca.transform(x_valid)
        else:
            new_x_valid_supervised = None

        # Remove items from n_comp_list that are outside the bounds
        max_n_comp = new_x_train_supervised.shape[1]
        n_comp_possible = [el for el in n_comp_list if el < max_n_comp]
        n_comp_list = n_comp_possible + [max_n_comp]

        print "Saving embeddings"
        for n_comp in n_comp_list:
            file_name = save_path + '/rpca_' + str(n_comp) + '_embedding.npz'
            np.savez(file_name,
                     x_train_supervised=new_x_train_supervised[:, :n_comp],
                     y_train_supervised=y_train,
                     x_valid_supervised=new_x_valid_supervised[:, :n_comp],
                     y_valid_supervised=y_valid,
                     x_test_supervised=new_x_test_supervised[:, :n_comp],
                     y_test_supervised=y_test)

    elif method == "truncSVD":
        # Apply truncated svd (pca) for each nber of clusters
        for n_cl in n_comp_list:
            start_time = time.time()
            print "tsvd %d of out %d" % (n_cl, len(n_comp_list))
            tsvd = TruncatedSVD(n_components=n_cl)
            tsvd.fit(unsupervised)
            new_x_train_supervised = tsvd.transform(x_train)
            new_x_test_supervised = tsvd.transform(x_test)

            if dataset != 'opensnp':
                new_x_valid_supervised = tsvd.transform(x_valid)
            else:
                new_x_valid_supervised = None

            # Save embeddings
            file_name = save_path + '/tsvd_' + str(n_cl) + '_embedding.npz'
            np.savez(file_name, x_train_supervised=new_x_train_supervised,
                     y_train_supervised=y_train,
                     x_valid_supervised=new_x_valid_supervised[:, :n_comp],
                     y_valid_supervised=y_valid,
                     x_test_supervised=new_x_test_supervised,
                     y_test_supervised=y_test)
            print "... took %f s" % (time.time() - start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""PCA embedding""")

    parser.add_argument('-d',
                        type=str,
                        default="1000_genomes",
                        help='dataset')
    parser.add_argument('-save_path',
                        '-sp',
                        default='/data/lisatmp4/romerosa/feature_selection/',
                        help='number of components for embedding')

    args = parser.parse_args()

    pca(args.d, [1, 2, 5, 10, 20, 50,
                 100, 200, 400, 600, 800,
                 1000, 1200, 1400, 1600, 2000, 2400, 2600, 3000],
        args.save_path)
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
