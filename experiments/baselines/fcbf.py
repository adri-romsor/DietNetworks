from experiments.common.dorothea import load_data
from aggregate_dataset import load_data23andme_baselines

from skfeature.function.information_theoretical_based import FCBF

import argparse
import numpy as np

import os

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

def fcbf_23andme(dataset, n_comp, save_path):
    # Load data
    print "Loading data"
    if dataset == "opensnp":
        train_supervised, test_supervised, unsupervised = \
            load_data23andme_baselines()
    else:
        raise ValueError("Unknown dataset")

    # Save embeddings for supervised training
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print 'Shapes'
    print train_supervised
    print type(train_supervised)
    print train_supervised[0].shape
    print train_supervised[1].shape

    idx = FCBF.fcbf(
            train_supervised[0],
            train_supervised[1],
            n_selected_features=n_comp)

    print "Detailed information while running fcbf"
    print "Selected indices: {}".format(idx)
    print "Number of selected indices: {}".format(len(idx))
    print "x_train shape: {}".format(x_train.shape)

    new_x_train_supervised = train_supervised[0][:,idx[0:n_comp]]
    new_x_test_supervised = test_supervised[0][:,idx[0:n_comp]]
    #new_x_valid = x_valid[:,idx[0:n_comp]]

    file_name = save_path + 'fcbf_' + str(len(idx)) + '_embedding.npz'
    np.savez(
            file_name,
            x_train_supervised=new_x_train_supervised,
            y_train_supervised=train_supervised[1],
            x_test_supervised=new_x_test_supervised,
            y_test_supervised= test_supervised[1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""FCBF feature selection""")

    parser.add_argument('-d',
                        type=str,
                        default="opensnp",
                        help='dataset')

    parser.add_argument('-nc',
                        type=int,
                        default=100,
                        help='number of selected components')

    parser.add_argument('-save_path',
                        '-sp',
                        default='/Tmp/sylvaint/',
                        help='save path for the dataset')

    args = parser.parse_args()

    #fcbf_dorothea(args.nc, args.save_path)
    fcbf_23andme(args.d,args.nc, args.save_path)
