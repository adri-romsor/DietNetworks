import sys
sys.path.append('/data/lisatmp4/dejoieti/')

from sklearn.decomposition import PCA
from feature_selection.experiments.common.dorothea import load_data
from feature_selection.config import path_dorothea

import numpy as np
import matplotlib.pyplot as plt

def pca_dorothea(n_comp=50):
    save_path = '/data/lisatmp4/dejoieti/feature_selection/'

    x_train, y_train = load_data('train', return_format='numpy')
    x_valid, y_valid = load_data('valid', return_format='numpy')
    pca = PCA(n_components=n_comp)
    pca.fit(x_train)
    new_x_train = pca.transform(x_train)
    new_x_valid = pca.transform(x_valid)

    file_name = save_path + 'pca_' + str(n_comp) + '_embedding.npz'
    np.savez(file_name, x_train=new_x_train, y_train=y_train,
                        x_valid=new_x_valid, y_valid=y_valid)

    return pca


if __name__ == '__main__':
    pca_dorothea(n_comp=100)



    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
