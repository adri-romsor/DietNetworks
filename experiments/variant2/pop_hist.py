import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from getpass import getuser

from feature_selection.experiments.common import dataset_utils

def plot(dataset, which_set):

    if dataset == '1000_genomes':
        data = dataset_utils.load_1000_genomes(transpose=False,
                                               label_splits=[.75])
    else:
        print("Unknown dataset")
        return

    (_, y_train), (_, y_valid), (_, y_test), _ = data

    if which_set == 'train':
        y = y_train
    elif which_set == 'valid':
        y = y_valid
    elif which_set == 'test':
        y = y_test
    elif which_set == 'tot':
        y = np.concatenate((y_train, y_valid, y_test), axis=0)

    y = y.argmax(1)

    y_bin = np.bincount(y)

    labels = ['ACB', 'ASW', 'BEB', 'CDX', 'CEU', 'CHB', 'CHS', 'CLM', 'ESN',
              'FIN', 'GBR', 'GIH', 'GWD', 'IBS', 'ITU', 'JPT', 'KHV', 'LWK',
              'MSL', 'MXL', 'PEL', 'PJL', 'PUR', 'STU', 'TSI', 'YRI']

    x = range(26)
    x5 = [el+.5 for el in x]
    x10 = [el+.5 for el in x5]

    fig = plt.figure(figsize=(9, 9))
    # plt.clf()
    ax = fig.add_subplot(111)
    # ax.set_aspect(1)

    # fig,ax = plt.subplots()
    ax.bar(x5, height= y_bin)
    plt.xticks(x10, labels, rotation='vertical');

    ax.set_xlim(0, 27)

    plt.xlabel('Ethnicity')
    plt.ylabel('Number of subjects')

    plt.savefig('eth_histo.png', format='png')

    plt.show()




def main():
    parser = argparse.ArgumentParser(description='plot errors')
    parser.add_argument('-dataset',
                        default='1000_genomes',
                        help='Dataset')
    parser.add_argument('-which_set',
                        default='tot',
                        help='which set')
    args = parser.parse_args()

    plot(args.dataset, args.which_set)

if __name__ == "__main__":
    main()


# ['CHB', 'JPT', 'CHS', 'CDX', 'KHV', 'CEU', 'TSI', 'FIN', 'GBR', 'IBS', 'YRI', 'LWK', 'GWD', 'MSL', 'ESN', 'ASW', 'ACB', 'MXL', 'PUR', 'CLM', 'PEL', 'GIH', 'PJL', 'BEB', 'STU', 'ITU']
# ['ACB', 'ASW', 'BEB', 'CDX', 'CEU', 'CHB', 'CHS', 'CLM', 'ESN', 'FIN', 'GBR', 'GIH', 'GWD', 'IBS', 'ITU', 'JPT', 'KHV', 'LWK', 'MSL', 'MXL', 'PEL', 'PJL', 'PUR', 'STU', 'TSI', 'YRI']
