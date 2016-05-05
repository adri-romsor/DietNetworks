from experiments.common.dorothea import load_data

from skfeature.function.information_theoretical_based import FCBF

import argparse
import numpy as np

def fcbf_dorothea(n_comp, save_path):
    x_train, y_train = load_data('train', return_format='numpy')
    x_valid, y_valid = load_data('valid', return_format='numpy')

    idx = FCBF.fcbf(x_train, y_train, n_selected_features=n_comp)

    print "Detailed information while running fcbf"
    print "Selected indices: {}".format(idx)
    print "Number of selected indices: {}".format(len(idx))
    print "x_train shape: {}".format(x_train.shape)

    new_x_train = x_train[:,idx[0:n_comp]]
    new_x_valid = x_valid[:,idx[0:n_comp]]

    file_name = save_path + 'fcbf_' + str(len(idx)) + '_embedding.npz'
    np.savez(file_name, x_train=new_x_train, y_train=y_train,
             x_valid=new_x_valid, y_valid=y_valid)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""FCBF feature selection""")
    parser.add_argument('-nc',
                        type=int,
                        default=100,
                        help='number of selected components')

    parser.add_argument('-save_path',
                        '-sp',
                        default='/Tmp/sylvaint/',
                        help='save path for the dataset')

    args = parser.parse_args()

    fcbf_dorothea(args.nc, args.save_path)
