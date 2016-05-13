from sklearn.decomposition import RandomizedPCA, TruncatedSVD, PCA
# from experiments.common.dorothea import load_data
from aggregate_dataset import load_data23andme_baselines

import argparse
import os
import numpy as np
import time

# def pca_dorothea(n_comp, save_path):
#     x_train, y_train = load_data('train', return_format='numpy')
#     x_valid, y_valid = load_data('valid', return_format='numpy')
#     pca = PCA(n_components=n_comp)
#     pca.fit(x_train)
#     new_x_train = pca.transform(x_train)
#     new_x_valid = pca.transform(x_valid)

#     file_name = save_path + 'pca_' + str(n_comp) + '_embedding.npz'
#     np.savez(file_name, x_train=new_x_train, y_train=y_train,
#              x_valid=new_x_valid, y_valid=y_valid)

#     return pca


def pca(dataset, n_comp_list, save_path, method="PCA", split=0.8,
        test_sep=False):

    # Load data
    print "Loading data"
    if dataset == "opensnp":
        train_supervised, test_supervised, unsupervised = \
            load_data23andme_baselines(split=split)
    else:
        raise ValueError("Unknown dataset")

    # Save embeddings for supervised training
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not test_sep:
        unsupervised = np.concatenate((unsupervised, test_supervised[0]),
                                      axis=0)

    if method == "PCA":
        print "Applying PCA"
        # Extract PCA from unsupervised data
        pca = PCA()
        pca.fit(unsupervised)
        # Apply PCA to supervised training data
        new_x_train_supervised = pca.transform(train_supervised[0])
        new_y_train = train_supervised[1]
        new_x_test_supervised = pca.transform(test_supervised[0])
        new_y_test = test_supervised[1]

        # Remove items from n_comp_list that are outside the bounds
        max_n_comp = new_x_train_supervised.shape[1]
        n_comp_possible = [el for el in n_comp_list if el < max_n_comp]
        n_comp_list = n_comp_possible + [max_n_comp]

        print "Saving embeddings"
        for n_comp in n_comp_list:
            file_name = save_path + 'pca_' + str(n_comp) + '_embedding.npz'
            np.savez(file_name,
                     x_train_supervised=new_x_train_supervised[:, :n_comp],
                     y_train_supervised=new_y_train,
                     x_test_supervised=new_x_test_supervised[:, :n_comp],
                     y_test_supervised=new_y_test)

    elif method == "randPCA":
        print "Applying PCA"
        # Extract PCA from unsupervised data
        pca = RandomizedPCA()
        pca.fit(unsupervised)
        # Apply PCA to supervised training data
        new_x_train_supervised = pca.transform(train_supervised[0])
        new_x_test_supervised = pca.transform(test_supervised[0])

        # Remove items from n_comp_list that are outside the bounds
        max_n_comp = new_x_train_supervised.shape[1]
        n_comp_possible = [el for el in n_comp_list if el < max_n_comp]
        n_comp_list = n_comp_possible + [max_n_comp]

        print "Saving embeddings"
        for n_comp in n_comp_list:
            file_name = save_path + 'rpca_' + str(n_comp) + '_embedding.npz'
            np.savez(file_name,
                     x_train_supervised=new_x_train_supervised[:, :n_comp],
                     y_train_supervised=train_supervised[1],
                     x_test_supervised=new_x_test_supervised[:, :n_comp],
                     y_test_supervised=test_supervised[1])

    elif method == "truncSVD":
        # Apply truncated svd (pca) for each nber of clusters
        for n_cl in n_comp_list:
            start_time = time.time()
            print "tsvd %d of out %d" % (n_cl, len(n_comp_list))
            tsvd = TruncatedSVD(n_components=n_cl)
            tsvd.fit(unsupervised)
            new_x_train_supervised = tsvd.transform(train_supervised[0])
            new_x_test_supervised = tsvd.transform(test_supervised[0])

            # Save embeddings
            file_name = save_path + 'kmeans_' + str(n_cl) + '_embedding.npz'
            np.savez(file_name, x_train_supervised=new_x_train_supervised,
                     y_train_supervised=train_supervised[1],
                     x_test_supervised=new_x_test_supervised,
                     y_test_supervised=test_supervised[1])
            print "... took %f s" % (time.time() - start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""PCA embedding""")

    parser.add_argument('-d',
                        type=str,
                        default="opensnp",
                        help='dataset')
    parser.add_argument('-save_path',
                        '-sp',
                        default='/data/lisatmp4/romerosa/feature_selection/with_test/',
                        help='number of components for embedding')

    args = parser.parse_args()

    pca(args.d, [1, 2, 5, 10, 20, 50,
                 100, 200, 400, 600, 800,
                 1000, 1200, 1400, 1600, 2000],
        args.save_path)
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
