import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from getpass import getuser

from DietNetworks.experiments.common import dataset_utils

import model_helpers as mh

def plot(dataset, which_set, continent=True):

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

    if continent:
        continent_cat = mh.create_1000_genomes_continent_labels()
        y_cont = np.zeros(y.shape)

        for i,c in enumerate(continent_cat):
            for el in c:
                y_cont[y == el] = i

        print y_cont
        y = y_cont.astype('int32')
        labels = ['EAS', 'EUR', 'AFR', 'AMR', 'SAS']
        x = range(5)
        x5 = [el+.5 for el in x]
        x10 = [el+.5 for el in x5]

        xlabel = 'Geographical Region'
        max_x = 6
        filename = 'cont_histo.png'
    else:
        labels = ['ACB', 'ASW', 'BEB', 'CDX', 'CEU', 'CHB', 'CHS', 'CLM', 'ESN',
                  'FIN', 'GBR', 'GIH', 'GWD', 'IBS', 'ITU', 'JPT', 'KHV', 'LWK',
                  'MSL', 'MXL', 'PEL', 'PJL', 'PUR', 'STU', 'TSI', 'YRI']
        x = range(26)
        x5 = [el+.5 for el in x]
        x10 = [el+.5 for el in x5]

        xlabel = 'Ethnicity'
        max_x = 27

        filename = 'eth_histo.png'

    y_bin = np.bincount(y)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)

    ax.bar(x5, height= y_bin)
    plt.xticks(x10, labels, rotation='vertical', fontsize=24);

    ax.set_xlim(0, max_x)

    plt.xlabel(xlabel, fontsize=32)
    plt.ylabel('Number of subjects', fontsize=32)
    plt.tight_layout()

    plt.savefig(filename, format='png')

    plt.show()




def main():
    parser = argparse.ArgumentParser(description='Histograms.')
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
