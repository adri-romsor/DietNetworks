from sklearn.cluster import KMeans
# from experiments.common.dorothea import load_data
from aggregate_dataset import load_data23andme_baselines

import argparse
import numpy as np
import os
import time
import matplotlib.pyplot as plt


# def kmeans_dorothea(n_cl, save_path):
#     save_path = '/data/lisatmp4/dejoieti/feature_selection/'

#     x_train, y_train = load_data('train', return_format='numpy')
#     x_valid, y_valid = load_data('valid', return_format='numpy')
#     km = KMeans(n_clusters=n_cl)
#     km.fit(x_train)
#     new_x_train = km.transform(x_train)
#     new_x_valid = km.transform(x_valid)

#     file_name = save_path + 'kmeans_' + str(n_cl) + '_embedding.npz'
#     np.savez(file_name, x_train=new_x_train, y_train=y_train,
#              x_valid=new_x_valid, y_valid=y_valid)

#     return new_x_train, km

def kmeans(dataset, n_cl_list, save_path):
    # Load data
    print "Loading data"
    if dataset == "opensnp":
        train_supervised, test_supervised, unsupervised = \
            load_data23andme_baselines()
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
        new_x_train_supervised = km.transform(train_supervised[0])
        new_x_test_supervised = km.transform(test_supervised[0])

        # Save embeddings
        file_name = save_path + 'kmeans_' + str(n_cl) + '_embedding.npz'
        np.savez(file_name, x_train_supervised=new_x_train_supervised,
                 y_train_supervised=train_supervised[1],
                 x_test_supervised=new_x_test_supervised,
                 y_test_supervised=test_supervised[1],
                 kmeans=km)
        print "... took %f s" % (time.time() - start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Kmeans embedding""")

    parser.add_argument('-save_path',
                        '-sp',
                        default='/data/lisatmp4/romerosa/feature_selection/',
                        help='number of components for embedding')

    args = parser.parse_args()

    new_x, kmeans = kmeans("opensnp", [1, 2, 5, 10, 20, 50,
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
