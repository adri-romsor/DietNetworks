from sklearn.cluster import KMeans
from aggregate_dataset import load_data23andme_baselines
from feature_selection.experiments.common import dataset_utils, imdb

import argparse
import numpy as np
import os
import time
import matplotlib.pyplot as plt


def kmeans(dataset, n_cl_list, save_path):
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

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Apply k-means for each nber of clusters
    for n_cl in n_cl_list:
        start_time = time.time()
        print "kmeans %d of out %d" % (n_cl, len(n_cl_list))
        km = KMeans(n_clusters=n_cl)
        km.fit(unsupervised)
        new_x_train_supervised = km.transform(x_train)

        if dataset != 'opensnp':
            new_x_valid_supervised = km.transform(x_valid)
        else:
            new_x_valid_supervised = None

        new_x_test_supervised = km.transform(x_test)

        # Save embeddings
        file_name = save_path + '/kmeans_' + str(n_cl) + '_embedding.npz'
        np.savez(file_name, x_train_supervised=new_x_train_supervised,
                 y_train_supervised=y_train,
                 x_valid_supervised=new_x_valid_supervised,
                 y_valid_supervised=y_valid,
                 x_test_supervised=new_x_test_supervised,
                 y_test_supervised=y_test,
                 kmeans=km)
        print "... took %f s" % (time.time() - start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Kmeans embedding""")

    parser.add_argument('-save_path',
                        '-sp',
                        default='/data/lisatmp4/romerosa/feature_selection/',
                        help='number of components for embedding')

    args = parser.parse_args()

    new_x, kmeans = kmeans("1000_genomes", [1, 2, 5, 10, 20, 50,
                                            100, 200, 500, 1000,
                                            500, 5000, 10000,
                                            50000, 100000, 500000, 1000000],
                           args.save_path)

    # The following is testig how accurate are the clusters to guess y

    # x_train, y_train = load_data('train', return_format='numpy')
    # x_valid, y_valid = load_data('valid', return_format='numpy')
    # train_x_predict = kmeans.predict(x_train)
    # valid_x_predict = kmeans.predict(x_valid)
    #
    # y_predict = [np.mean(y_train[train_x_predict == i])
    #              for i in valid_x_predict]
    #
    # print(np.mean((y_predict - y_valid)**2))
    #
    # y_predict_hard = [0 if y < 0.5 else 1 for y in y_predict]
    #
    # print(np.mean((y_predict_hard - y_valid)**2))
