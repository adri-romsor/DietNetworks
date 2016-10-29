#from sklearn.decomposition import PCA
from experiments.common.dorothea import load_data

#from skfeature.function.information_theoretical_based import FCBF
from skfeature.utility.mutual_information import su_calculation, information_gain
import skfeature.utility.entropy_estimators as ee
import numpy as np

import argparse
import numpy as np

def information_gain_calculation(X,y):
    features = X.shape[1]
    i_g_results = []
    for i in range(features):
        i_g_results.append(information_gain(X[:,i],y))

    return i_g_results

def information_gain_dorothea(n_comp, save_path):
    x_train, y_train = load_data('train', return_format='numpy')
    x_valid, y_valid = load_data('valid', return_format='numpy')
    #pca = PCA(n_components=n_comp)
    #pca.fit(x_train)
    #new_x_train = pca.transform(x_train)
    #new_x_valid = pca.transform(x_valid)

    ig_idx = information_gain_calculation(x_train,y_train)
    values = ig_idx[:]
    #print "ig_idx: {}".format(ig_idx)
    ig_idx = np.argsort(ig_idx)
    ig_idx = ig_idx[::-1]
    print "Example of sorted array, first elt: {}".format(values[ig_idx[0]])

    ig_idx = ig_idx[:n_comp]

    new_x_train = x_train[:,ig_idx[0:n_comp]]
    new_x_valid = x_valid[:,ig_idx[0:n_comp]]

    print "shapes"
    print x_train.shape
    print new_x_train.shape

    print x_valid.shape
    print new_x_valid.shape

    file_name = save_path + 'information_gain_' + str(n_comp) + '_embedding.npz'
    np.savez(file_name, x_train=new_x_train, y_train=y_train,
             x_valid=new_x_valid, y_valid=y_valid)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Information gain""")
    parser.add_argument('-nc',
                        type=int,
                        default=100,
                        help='number of components for embedding')

    parser.add_argument('-save_path',
                        '-sp',
                        default='/Tmp/sylvaint/',
                        help='save path for the dataset')

    args = parser.parse_args()

    information_gain_dorothea(args.nc, args.save_path)
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
