from sklearn.decomposition import PCA
from experiments.common.dorothea import load_data

import numpy as np
import matplotlib.pyplot as plt

def pca_dorothea(x, nc=None):

    pca = PCA(n_components=nc)
    pca.fit(x)

    return pca.transform(x), pca


if __name__ == '__main__':
    x, y = load_data('train', return_format='numpy')
    new_x, pca = pca_dorothea(x, nc=50)

    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
