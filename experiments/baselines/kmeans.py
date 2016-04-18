from sklearn.cluster import KMeans
from experiments.common.dorothea import load_data

import numpy as np
import matplotlib.pyplot as plt


def kmeans_dorothea(x, nc=50):

    km = KMeans(verbose=True, n_clusters=nc)
    km.fit(x)

    return km.transform(x), km


if __name__ != '__main__':
    x_train, y_train = load_data('train', return_format='numpy')
    new_x, kmeans = kmeans_dorothea(x_train, nc=50)


    # The following is testig how accurate are the clusters to guess y

    x_valid, y_valid = load_data('valid', return_format='numpy')
    train_x_predict = kmeans.predict(x_train)
    valid_x_predict = kmeans.predict(x_valid)

    y_predict = [np.mean(y_train[train_x_predict == i])
                 for i in valid_x_predict]

    print(np.mean((y_predict - y_valid)**2))

    y_predict_hard = [0 if y < 0.5 else 1 for y in y_predict]

    print(np.mean((y_predict_hard - y_valid)**2))
