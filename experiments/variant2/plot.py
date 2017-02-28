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

        error_var = np.load(os.path.join(file_path, 'errors_supervised_last.npz'))

        train_loss = error_var['arr_0'][-2]
        train_acc = error_var['arr_0'][-1]
        valid_loss = error_var['arr_1'][-2]
        valid_acc = error_var['arr_1'][-1]

        max_epoch = len(train_loss)
        epochs = range(max_epoch)

        if metric == 'loss':
            plt.plot(epochs, train_loss, '-'+c, label='train_loss: '+m)
            plt.plot(epochs, valid_loss, '--'+c, label='val_loss: '+m)
        elif metric == 'acc':
            plt.plot(epochs, train_acc, '-'+c, label='train_acc: '+m)
            plt.plot(epochs, valid_acc, '--'+c, label='val_acc: '+m)
        else:
            raise ValueError('Unknown metric')

    plt.legend(loc=1)
    plt.ylabel(metric)
    plt.xlabel('Epochs')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot errors.')
    parser.add_argument('-dataset',
                        default='1000_genomes',
                        help='Dataset')
    parser.add_argument('-metric',
                        default='acc',
                        help='Metric (acc or loss)')
    parser.add_argument('-path',
                        default='/data/lisatmp4/romerosa/DietNetworks/',
                        help='Path to errors file')
    parser.add_argument('-models',
                        type=list,
                        default=[
    'dietnets_final_unsupervisedhist3x26fold0_final_our_model1.0_histo3x26_lr-0.0001_anneal-0.99_eni-0.01_dni-0.01_accuracy_Ri10.0_hu-100_tenc-100_tdec-100_hs-100_fold0'
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
