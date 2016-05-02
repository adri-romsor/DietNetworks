from sklearn.decomposition import PCA
from experiments.common.dorothea import load_data

import argparse
import numpy as np

def pca_dorothea(n_comp, save_path):
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
    parser = argparse.ArgumentParser(description="""PCA embedding""")
    parser.add_argument('-nc',
                        type=int,
                        default=100,
                        help='number of components for embedding')

    parser.add_argument('-save_path',
                        '-sp',
                        default='/data/lisatmp4/dejoieti/feature_selection/',
                        help='number of components for embedding')

    args = parser.parse_args()

    pca_dorothea(args.nc, args.save_path)
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
