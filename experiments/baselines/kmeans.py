import sys
sys.path.append('/data/lisatmp4/dejoieti/')

from sklearn.cluster import KMeans
from feature_selection.experiments.common.dorothea import load_data

import numpy as np
import matplotlib.pyplot as plt


def kmeans_dorothea(n_cl=50):
    save_path = '/data/lisatmp4/dejoieti/feature_selection/'

    x_train, y_train = load_data('train', return_format='numpy')
    x_valid, y_valid = load_data('valid', return_format='numpy')
    km = KMeans(n_clusters=n_cl)
    km.fit(x_train)
    new_x_train = km.transform(x_train)
    new_x_valid = km.transform(x_valid)

    file_name = save_path + 'kmeans_' + str(n_cl) + '_embedding.npz'
    np.savez(file_name, x_train=new_x_train, y_train=y_train,
                        x_valid=new_x_valid, y_valid=y_valid)

    return new_x_train, km


if __name__ == '__main__':
    x_train, y_train = load_data('train', return_format='numpy')
    new_x, kmeans = kmeans_dorothea(n_cl=50)


    # The following is testig how accurate are the clusters to guess y

    x_valid, y_valid = load_data('valid', return_format='numpy')
    train_x_predict = kmeans.predict(x_train)
    valid_x_predict = kmeans.predict(x_valid)

    y_predict = [np.mean(y_train[train_x_predict == i])
                 for i in valid_x_predict]

    print(np.mean((y_predict - y_valid)**2))

    y_predict_hard = [0 if y < 0.5 else 1 for y in y_predict]

    print(np.mean((y_predict_hard - y_valid)**2))
