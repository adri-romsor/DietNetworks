import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from getpass import getuser


def plot(dataset,
         metric='loss',
         model_path=None,
         models=None,
         colors=None):

    for (m, c) in zip(models, colors):
        file_path = os.path.join(model_path, dataset, m)
        if not os.path.exists(file_path):
            raise ValueError('The path to {} does not exist'.format(file_path))

        error_var = np.load(os.path.join(file_path, 'errors_supervised.npz'))

        if 'our_model' in m:
            train_loss = error_var['arr_0']
            train_acc = []
            valid_loss = error_var['arr_1']
            vaid_acc = error_var['arr_3']
        else:
            train_loss = error_var['arr_0']
            train_acc = error_var['arr_2']

            valid_loss = error_var['arr_4']
            valid_acc = error_var['arr_6']

        max_epoch = len(train_loss)
        epochs = range(max_epoch)

        if metric == 'loss':
            plt.plot(epochs, train_loss, '-'+c, label='train_loss: '+m)
            plt.plot(epochs, valid_loss, '--'+c, label='val_loss: '+m)
            l=1
        elif metric == 'acc':
            plt.plot(epochs, train_acc, '-'+c, label='train_acc: '+m)
            plt.plot(epochs, valid_acc, '--'+c, label='val_acc: '+m)
            l=0
        else:
            raise ValueError('Unknown metric')

    plt.legend(loc=l)
    plt.ylabel(metric)
    plt.xlabel('Epochs')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='plot errors')
    parser.add_argument('-dataset',
                        default='1000_genomes',
                        help='Dataset')
    parser.add_argument('-metric',
                        default='loss',
                        help='Metric (acc or loss)')
    parser.add_argument('-path',
                        default='/data/lisatmp4/romerosa/feature_selection/',
                        help='Path to errors file')
    parser.add_argument('-models',
                        type=list,
                        default=[
                            'basic_1.0_sup/',
                            'basic_1.0_sup_unsup/',
                            # 'our_model_sup/',
                            # 'our_model_sup_unsup/'
                            ],
                        help='List of model names.')
    parser.add_argument('-colors',
                        type=list,
                        default=['r', 'g', 'b', 'k'],
                        help='Colors to plot curves.')
    args = parser.parse_args()

    plot(args.dataset, args.metric, args.path, args.models, args.colors)

if __name__ == "__main__":
    main()
